import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings
import textwrap

from dotenv import load_dotenv
import os
from mistralai import Mistral

import streamlit as st

model = "mistral-large-latest"

load_dotenv(dotenv_path = "dot.env")
gemini_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")

from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")
dimensions = 512

@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions)
    genai.configure(api_key=gemini_api_key)
    model_1 = genai.GenerativeModel("gemini-1.5-flash")
    client = Mistral(api_key=mistral_api_key)
    return client, model_1, embedding_model, cross_encoder

@st.cache_data
def load_data():
    text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df (3).csv")
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    embeddings = torch.tensor(np.stack(text_chunks_and_embedding_df["embedding"].tolist(), axis = 0),dtype = torch.float32)
    markdown_text_and_chunks = text_chunks_and_embedding_df.to_dict(orient = "records")
    return embeddings, markdown_text_and_chunks

client, model_1, embedding_model, cross_encoder = load_resources()
embeddings, markdown_text_and_chunks = load_data()

def get_unique_numbers(lists):
    all_numbers = set([num for sublist in lists for num in sublist])
    return sorted(list(all_numbers))
def get_unique_numbers_optimized(lists):
    return sorted(set().union(*[set(sublist) for sublist in lists]))

def split_text_to_sentences(text):
    sentences = [sentence.strip() for sentence in text.split('\n') if sentence.strip()]
    return sentences

def query_expander(query,n=4):
  base_template_prompt = f"""You are an AI language model assistant. Your task is to generate {n} different versions of the
  given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question and adding new keywords
  similar in meaning to the ones used by the user, your goal is to help the user overcome some of the limitations of the cosine-based similarity search.
  Provide these alternative questions separated by newlines.I just want the alternative queries nothing else seperated by new lines... DON'T ADD ANYTHING
  ELSE. Original question: {query}"""
  output = model_1.generate_content(base_template_prompt,
    generation_config=genai.types.GenerationConfig(
        temperature=2.0,
    ))
  return split_text_to_sentences(output.text)

def retrieve_relevant_resources(
    query: str,
    embeddings : torch.tensor,
    model: SentenceTransformer=embedding_model,
    n_resources_to_return : int = 15,
    cross_encoder_model: SentenceTransformer=cross_encoder,
    print_time: bool = True):
  """
  Embeds a query within a model and returns top k scores and indices from the embeddings
  """
  #Expand the query and get more similar queries like the one that has been asked by the user.
  queries = query_expander(query)
  #Add the original query to the list of the queries that you are going to iterate over
  queries.append(query)
  indices_values = []
  #Get the indices values for those new queries as well
  for index, query in enumerate(queries):
    query_encoded = embedding_model.encode(query)
    similarities_1 = cos_sim(query_encoded, embeddings)
    topk_result = torch.topk(similarities_1, 15)
    indices = topk_result.indices.squeeze().tolist()
    indices_values.append(indices)

  final_indices = get_unique_numbers_optimized(indices_values)

  #Get the passages corresponding to the final_indices values
  passages = []
  for i in range(len(final_indices)):
    document = markdown_text_and_chunks[final_indices[i]]["processed_chunk_text"]
    document = str(document)
    passages.append(document)
  results = cross_encoder_model.rank(query, passages, return_documents=True, top_k=7)
  x = []
  for i in range(len(results)):
    x.append(results[i]["corpus_id"])
  final_indices_to_be_sent = [final_indices[n] for n in x]
  return topk_result.values, final_indices_to_be_sent


from contextlib import contextmanager
def prompt_formatter(query,context_items):
  context = "\n\n".join([item["chunk"] for item in context_items])

  prompt = f"""
You are JIIT Assistant, the official AI helper for Jaypee Institute of Information Technology (JIIT). Your role is to provide accurate, helpful, and professional responses to queries about JIIT for students, parents, and visitors. You should reflect the institution's values of academic excellence, innovation, and holistic development.

## Key Behaviors:
- Maintain a professional, warm, and helpful tone.
- Provide accurate, up-to-date information. If unsure, acknowledge limitations and direct users to university contacts.
- Use concise, clear language suitable for both students and parents.
- Protect sensitive information and uphold JIIT's reputation as a premier technical institution.

## Response Format:
1. Direct, informative answers for general queries.
2. Step-by-step guidance for procedural questions.
3. Always include relevant contact details or links when necessary.

## Example Queries:

**Academic Program Query:**
User: What B.Tech programs does JIIT offer?
Assistant: JIIT offers B.Tech programs in:
- Computer Science and Engineering (CSE)
- Information Technology (IT)
- Electronics and Communication Engineering (ECE)
- Biotechnology (BT)

Each program is industry-focused with hands-on training. For detailed curriculum and career prospects, visit www.jiit.ac.in or contact admissions at admission@jiit.ac.in.

**Admission Process Query:**
User: How can I apply for admission?
Assistant: Here's how to apply:
1. JEE Main scores are required for B.Tech admissions.
2. Management quota seats are also available.
3. Fill out the application at www.jiit.ac.in and submit documents like Class 10th and 12th marksheets and JEE score card.

Contact admissions at admission@jiit.ac.in or call [Insert Number] for more help.

**Campus Life Query:**
User: What facilities are available on campus?
Assistant: JIIT offers comprehensive academic and student facilities:
- Multimedia-equipped classrooms and state-of-the-art labs.
- Wi-Fi-enabled campus and a central library.
- Sports facilities, food courts, medical center, and hostels.

For more details, ask about specific facilities or visit www.jiit.ac.in.

## General Guidelines:
- Acknowledge complaints professionally and provide contacts for further assistance.
- Highlight JIIT's rankings and notable achievements when relevant.
- Direct users to the IT helpdesk for technical issues.

Context: {context}
Use ONLY the context above. If information is insufficient, acknowledge this and suggest where the user might find more details.

Query: {query}
Answer: """

  return prompt
def run_mistral(user_message, model="mistral-large-latest"):
    messages = [
        {
            "role": "user", "content": user_message
        }
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return (chat_response.choices[0].message.content)

def Mistral_Chatbot(query):
  scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings)
  context_items = [markdown_text_and_chunks[idx] for idx in indices]
  prompt = prompt_formatter(query,context_items)
  return run_mistral(prompt)

st.title("JIIT Assistant")

user_input = st.text_input("Ask a question about JIIT:", "")

if user_input:
    response = Mistral_Chatbot(user_input)
    st.write(response)
