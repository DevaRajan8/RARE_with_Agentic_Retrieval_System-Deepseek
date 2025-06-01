# RARE with Agentic Retrieval Systems

A comprehensive framework combining two powerful Streamlit applications for retrieval-augmented generation: the **Enhanced RAG with Chain-of-Thought Reasoning (RARE)** system and the **Agentic Retrieval System**. Together, they offer robust document processing, intelligent retrieval, and advanced response generation capabilities powered by the Groq API.

## 🌟 Overview

This repository contains two complementary systems that work together to provide state-of-the-art document retrieval and question-answering capabilities:

- **RARE System**: Foundation RAG with chain-of-thought reasoning
- **Agentic Retrieval System**: Advanced multi-strategy retrieval with intelligent routing

Both systems leverage vector databases for efficient storage, semantic similarity searches, and the Groq API for generating high-quality responses.

## 🧠 RARE-Enhanced RAG with Chain-of-Thought Reasoning

### Overview
The RARE system is a retrieval-augmented generation application designed to provide intelligent, reasoned answers based on a knowledge base of uploaded documents. It utilizes ChromaDB for efficient storage and retrieval of document embeddings, enabling semantic similarity searches with a systematic chain-of-thought reasoning approach.

### Key Features
- **Multi-format Document Support**: Processes TXT, PDF, DOCX, CSV, JSON, and Excel files
- **Vector Database**: Uses ChromaDB for persistent storage and fast retrieval of document embeddings
- **Semantic Search**: Retrieves documents based on cosine similarity of embeddings
- **Chain-of-Thought Reasoning**: Generates structured responses with a 6-step reasoning process
- **Persistent Knowledge Base**: Retains documents across sessions
- **User Interface**: Streamlit-based interface for document management and querying

### How RARE Works

1. **Document Upload**: Users upload files or manually add text
2. **Text Extraction**: Extracts content from supported file formats
3. **Embedding Generation**: Creates embeddings using the `all-MiniLM-L6-v2` SentenceTransformer model
4. **Storage**: Stores embeddings and metadata in ChromaDB
5. **Querying**: Retrieves top-k relevant documents based on query embedding similarity
6. **Response Generation**: Constructs a chain-of-thought prompt with retrieved documents and generates a response via the Groq API

### Chain-of-Thought Process
1. **Understanding**: Analyze and restate the user's question
2. **Context Analysis**: Examine retrieved documents for relevance
3. **Reasoning Process**: Break down complex questions step-by-step
4. **Evidence Synthesis**: Combine information from multiple sources
5. **Conclusion Formation**: Provide well-reasoned answers
6. **Verification**: Double-check answer consistency and completeness

### Usage - RARE System
- **Upload Documents**: Add files via the uploader (supports multiple formats)
- **Add Manually**: Input text and sources directly
- **Clear Knowledge Base**: Remove all stored documents
- **Ask Questions**: Query the system to receive reasoned responses

---

## 🤖 Agentic Retrieval System

### Overview
The Agentic Retrieval System builds upon the retrieval-augmented generation concept by introducing advanced, agentic retrieval strategies. It supports multiple knowledge bases and dynamically selects the best retrieval mode for a query, offering more flexible and intelligent retrieval capabilities than traditional RAG systems.

### Key Features
- **Multiple Knowledge Bases**: Organize documents into distinct knowledge bases with descriptions
- **Intelligent Routing**: Routes queries to the most relevant knowledge base(s) using the Groq API
- **Dynamic Retrieval Modes**: Supports multiple retrieval strategies:
  - `chunks`: Retrieves specific document excerpts
  - `files_via_metadata`: Retrieves files based on metadata matches
  - `files_via_content`: Retrieves files based on full content similarity
  - `web_search`: Simulates external information retrieval (mock implementation)
- **Iterative Refinement**: Adjusts retrieval strategies if initial results are insufficient
- **Response Generation**: Generates answers from retrieved context using the Groq API
- **Supported Formats**: Handles PDF, DOCX, and TXT files

### How Agentic Retrieval Works

1. **Knowledge Base Management**: Create or select knowledge bases and upload documents
2. **Document Processing**: Extracts text, creates chunks, and generates embeddings
3. **Query Routing**: Identifies relevant knowledge bases for the query
4. **Mode Selection**: Determines the optimal retrieval mode (e.g., chunks, files_via_content)
5. **Retrieval**: Executes retrieval based on the selected mode
6. **Result Evaluation**: Assesses result sufficiency, refining with alternative modes if needed
7. **Response Generation**: Combines retrieved context into a coherent answer via the Groq API

### Usage - Agentic Retrieval System
- **Create Knowledge Base**: Define a new knowledge base with a name and description
- **Upload Documents**: Add files to a selected knowledge base
- **Query the System**: Enter a query to retrieve information and generate a response
- **View Knowledge Bases**: Inspect and manage existing knowledge bases

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Groq API key

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/DevaRajan8/RARE_with_Agentic_Retrieval_System-Deepseek
   cd RARE_with_Agentic_Retrieval_System-Deepseek
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**
   ```bash
   export GROQ_API_KEY=your_api_key_here
   ```
   
   Or create a `.env` file:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

4. **Run the Applications**
   
   **For RARE System:**
   ```bash
   streamlit run rare_app.py
   ```
   
   **For Agentic Retrieval System:**
   ```bash
   streamlit run agentic_retrieval_app.py
   ```

5. **Access the Applications**
   - Open your browser and navigate to `http://localhost:8501`

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit>=1.28.0
requests>=2.31.0
sentence-transformers>=2.2.2
chromadb>=0.4.0
numpy>=1.24.0
pandas>=2.0.0
PyPDF2>=3.0.1
python-docx>=0.8.11
groq>=0.4.0
pathlib
python-dotenv>=1.0.0
```

## ⚙️ Configuration

### API Configuration
- **Groq API Key**: Required for both systems. Set via the `GROQ_API_KEY` environment variable
- **Model**: Uses Groq's language models for response generation
- **Embedding Model**: Both systems use `all-MiniLM-L6-v2` from SentenceTransformers (configurable in code)

### Database Configuration
- **Vector Database**: RARE uses ChromaDB with persistent storage at `./chroma_db`
- **Storage**: Local persistent storage for embeddings and metadata

## 🛠️ Advanced Features

### Multi-Knowledge Base Management
The Agentic Retrieval System allows you to:
- Create multiple specialized knowledge bases
- Automatically route queries to relevant knowledge bases
- Manage documents across different domains or topics

### Intelligent Retrieval Strategies
- **Chunk-based Retrieval**: For precise information extraction
- **Metadata-based Retrieval**: For document-level filtering
- **Content-based Retrieval**: For semantic document matching
- **Hybrid Approaches**: Combining multiple strategies for optimal results

### Chain-of-Thought Integration
Both systems provide transparent reasoning processes:
- Step-by-step problem decomposition
- Evidence evaluation and synthesis
- Logical conclusion formation
- Answer verification and validation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact 
[Devarajan S](mailto:devarajan8.official@gmail.com) 
