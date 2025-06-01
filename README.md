Enhanced RAG and Agentic Retrieval Systems
This repository contains two powerful Streamlit applications for retrieval-augmented generation: the Enhanced RAG with Chain-of-Thought Reasoning (RARE) system and the Agentic Retrieval System. Together, they form a comprehensive framework referred to as "RARE with Agentic Retrieval," offering robust document processing, retrieval, and response generation capabilities. Both systems leverage the Groq API for generating responses and support various document formats, with the Agentic Retrieval System adding advanced retrieval strategies to complement the RARE foundation.

Enhanced RAG with Chain-of-Thought Reasoning (RARE)
Overview
The RARE system is a retrieval-augmented generation (RAG) application designed to provide intelligent, reasoned answers based on a knowledge base of uploaded documents. It utilizes a vector database (ChromaDB) for efficient storage and retrieval of document embeddings, enabling semantic similarity searches to find relevant documents for a given query. Responses are generated using the Groq API with a chain-of-thought reasoning approach, ensuring detailed and structured answers.
Key Features

Multi-format Document Support: Processes TXT, PDF, DOCX, CSV, JSON, and Excel files.
Vector Database: Uses ChromaDB for persistent storage and fast retrieval of document embeddings.
Semantic Search: Retrieves documents based on cosine similarity of embeddings.
Chain-of-Thought Reasoning: Generates structured responses with a step-by-step reasoning process.
Persistent Knowledge Base: Retains documents across sessions.
User Interface: Streamlit-based interface for document management and querying.

How It Works

Document Upload: Users upload files or manually add text.
Text Extraction: Extracts content from supported file formats.
Embedding Generation: Creates embeddings using the all-MiniLM-L6-v2 SentenceTransformer model.
Storage: Stores embeddings and metadata in ChromaDB.
Querying: Retrieves top-k relevant documents based on query embedding similarity.
Response Generation: Constructs a chain-of-thought prompt with retrieved documents and generates a response via the Groq API.

Usage

Upload Documents: Add files via the uploader (supports multiple formats).
Add Manually: Input text and sources directly.
Clear Knowledge Base: Remove all stored documents.
Ask Questions: Query the system to receive reasoned responses.


Agentic Retrieval System
Overview
The Agentic Retrieval System builds upon the retrieval-augmented generation concept by introducing advanced, agentic retrieval strategies. It supports multiple knowledge bases and dynamically selects the best retrieval mode for a query, including options like document chunks, metadata-based file retrieval, content-based file retrieval, and even mock web searches. This system enhances the RARE framework by offering more flexible and intelligent retrieval capabilities, also powered by the Groq API for response generation.
Key Features

Multiple Knowledge Bases: Organize documents into distinct knowledge bases with descriptions.
Intelligent Routing: Routes queries to the most relevant knowledge base(s) using the Groq API.
Dynamic Retrieval Modes: Supports:
chunks: Retrieves specific document excerpts.
files_via_metadata: Retrieves files based on metadata matches.
files_via_content: Retrieves files based on full content similarity.
web_search: Simulates external information retrieval (mock implementation).


Iterative Refinement: Adjusts retrieval strategies if initial results are insufficient.
Response Generation: Generates answers from retrieved context using the Groq API.
Supported Formats: Handles PDF, DOCX, and TXT files.

How It Works

Knowledge Base Management: Create or select knowledge bases and upload documents.
Document Processing: Extracts text, creates chunks, and generates embeddings.
Query Routing: Identifies relevant knowledge bases for the query.
Mode Selection: Determines the optimal retrieval mode (e.g., chunks, files_via_content).
Retrieval: Executes retrieval based on the selected mode.
Result Evaluation: Assesses result sufficiency, refining with alternative modes if needed.
Response Generation: Combines retrieved context into a coherent answer via the Groq API.

Usage

Create Knowledge Base: Define a new knowledge base with a name and description.
Upload Documents: Add files to a selected knowledge base.
Query the System: Enter a query to retrieve information and generate a response.
View Knowledge Bases: Inspect and manage existing knowledge bases.


Installation
To set up and run either system:

Clone the Repository:
git clone https://github.com/your-repo/rare-agentic-retrieval.git
cd rare-agentic-retrieval


Install Dependencies:
pip install -r requirements.txt

Ensure you have Python 3.8+ installed.

Set Environment Variables:

Set your Groq API key:export GROQ_API_KEY=your_api_key_here




Run the Application:

For RARE:streamlit run rare_app.py


For Agentic Retrieval:streamlit run agentic_retrieval_app.py






Requirements
Create a requirements.txt file with the following dependencies:
streamlit
requests
sentence-transformers
chromadb
numpy
pandas
PyPDF2
python-docx
groq

Install them using pip install -r requirements.txt.

Configuration

Groq API Key: Required for both systems. Set via the GROQ_API_KEY environment variable.
Embedding Model: Both use all-MiniLM-L6-v2 from SentenceTransformers (configurable in code).
Vector Database: RARE uses ChromaDB with persistent storage at ./chroma_db.


Examples
RARE System

Query: "What is the difference between AI and Machine Learning?"
Response: A detailed answer with reasoning steps, citing relevant documents (e.g., AI overview, ML pipeline).

Agentic Retrieval System

Query: "Summarize the main points from the financial report."
Response: A concise summary from the most relevant document, with source details and retrieval mode noted.


Limitations

Document Size: Large files may be truncated or chunked, potentially losing context.
Web Search: Agentic Retrieval’s web search is a mock implementation; real API integration is needed for production.
Performance: Processing many documents can be slow without optimization.


Contributing
Contributions are welcome! Please open an issue or submit a pull request with enhancements or fixes.

License
This project is licensed under the MIT License.
