# 📄 RAG-Enhanced Chat with PDF: Overview

This application is a **Production-Ready Retrieval-Augmented Generation (RAG)** chatbot designed to help users interact with PDF documents. Unlike basic RAG systems, it uses specialized strategies for different types of documents (Research Papers, Books, Receipts, etc.).

---

## 🚀 Core Idea
The chatbot allows you to upload a PDF, automatically (or manually) detect its category, and then ask questions. It "remembers" the document's content and provides accurate answers based on the retrieved context.

---

## 🛠️ Key Components & Structure

### 1. **Technology Stack**
- **Orchestration**: [LangChain](https://python.langchain.com/) (Handles the RAG flow).
- **Frontend**: Handled in a separate repository.
- **Vector Database**: [ChromaDB Cloud](https://www.trychroma.com/) (Stores document "memories").
- **LLM & Embeddings**: GPT-4o-mini and text-embedding-3-small (via [AI/ML API](https://aimlapi.com/)).

### 2. **File Structure (`src` directory)**
- `main.py`: The entry point. Handles the UI and file uploads.
- `agent.py`: The "brain" that coordinates between the user's question, the document content, and the AI.
- `document_processor.py`: Splits the PDF into intelligent segments (chunks) based on its type (e.g., chapters for books).
- `vector_store.py`: Saves and searches document segments in the cloud database.
- `llm_client.py` & `embeddings.py`: Wrappers for interacting with the AI models.
- `config.py`: Centralized settings for API keys and model parameters.

---

## 🌟 Why it's Special

### 1. **Category-Specific Chunking**
Most RAG systems split text into fixed-size blocks (e.g., every 500 characters). This app is smarter:
- **Research Papers**: Splits by sections like "Abstract" or "Methods".
- **Books**: Splits by chapters.
- **Receipts**: Extracts line items and totals.

### 2. **Agentic Reasoning**
If you ask a complex question, the `agent.py` can:
1. Break the question into smaller sub-tasks.
2. Search for information for each sub-task separately.
3. Combine the findings into one comprehensive answer.

### 3. **Hybrid Knowledge Retrieval**
- **Primary**: It mostly searches the uploaded PDF (85%).
- **Fallback**: If the PDF doesn't have the answer, it can perform a quick **Web Search** (15%) using DuckDuckGo to provide a complete response.

---

## 🔄 Simple Workflow
1. **Upload**: User uploads a PDF and selects a category (Research Paper, Book, etc.).
2. **Process**: The app extracts text, chunks it intelligently, and saves it as "embeddings" in ChromaDB Cloud.
3. **Question**: User types a question in the chat.
4. **Retrieve**: The agent searches the database for the most relevant parts of the document.
5. **Answer**: The LLM (GPT-4o-mini) generates a response using the retrieved context.
6. **Memory**: The app remembers previous parts of the conversation for follow-up questions.
