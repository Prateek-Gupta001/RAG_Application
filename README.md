# 🔍 Advanced RAG Application with Multi-Query Generation

A powerful Retrieval Augmented Generation system leveraging Mistral and Gemini APIs for enhanced document retrieval and response generation.

## ✨ Key Features

- 🤖 **Dual LLM Pipeline**: Gemini API handles query expansion while Mistral Large (latest) generates final responses

- 📚 **State-of-the-Art Embedding & Re-Ranking**: Utilizing MixedBread Embedding Model and MixedBread.ai's reranking model for optimal source selection

- 💾 **Lightweight Storage**: In-memory tensor database for minimal footprint compared to traditional solutions like Chroma/Pinecone - perfect for rapid prototyping and small/medium datasets

- 🎯 **Self-Querying Retrieval**: Gemini Flash API automatically generates multiple queries from user input to enhance document retrieval quality

- 🎨 **Clean UI**: Built with Streamlit for intuitive user experience

## 🔄 Workflow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '16px', 'fontFamily': 'arial' }}}%%
flowchart LR
    %% Define nodes with uniform rectangles
    A[["🔍 User Query"]] --> B[["⚡ Gemini API"]]
    B --> |Query Expansion| C[["📝 Queries (4x)"]]
    C --> D[["💾 Vector DB"]]
    D --> |Retrieve| E[["📚 Documents"]]
    E --> |Combine| F[["🔄 Assembly"]]
    F --> G[["🤖 Mistral LLM"]]
    G --> H[["✨ Response"]]

    %% Modern color scheme
    classDef default fill:#2D3748,stroke:#4A5568,stroke-width:3px,color:#FFFFFF,rx:10,ry:10
    classDef input fill:#5B21B6,stroke:#4C1D95,stroke-width:3px,color:#FFFFFF,rx:10,ry:10
    classDef api fill:#059669,stroke:#047857,stroke-width:3px,color:#FFFFFF,rx:10,ry:10
    classDef db fill:#9C4221,stroke:#7B341E,stroke-width:3px,color:#FFFFFF,rx:10,ry:10
    classDef output fill:#BE185D,stroke:#9D174D,stroke-width:3px,color:#FFFFFF,rx:10,ry:10

    %% Apply styles
    class A,H input
    class B,G api
    class D db
    class E output

    %% Adjust spacing and size
    linkStyle default stroke:#4A5568,stroke-width:2px

    %% Group related components
    subgraph Query Processing
        direction LR
        B
        C
    end
    
    subgraph Data Retrieval
        direction LR
        D
        E
    end
   
   subgraph Response Generation
       F
       G
   end
