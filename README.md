# RAG-Enhanced Chat with PDF

A production-ready Retrieval-Augmented Generation (RAG) chatbot for PDF documents with multi-category support, built using Langchain, ChromaDB Cloud, and modern AI APIs.

---

## 📚 Basic Idea

This application allows users to upload PDF documents and ask questions about their content using natural language. The system:

1. **Extracts** text from PDF documents
2. **Processes** content based on document category (Research Papers, Articles, Books, etc.)
3. **Stores** document chunks as vector embeddings in ChromaDB Cloud
4. **Retrieves** relevant context when users ask questions
5. **Generates** accurate answers using LLMs powered by the retrieved context

**Key Feature**: Multi-category document processing with specialized chunking strategies for different document types.

---

## 🚀 Why Our RAG is Different

### Traditional RAG Limitations:
- ❌ One-size-fits-all chunking (fixed 500 tokens)
- ❌ No document type awareness
- ❌ Poor handling of structured documents
- ❌ Loss of document hierarchy
- ❌ Manual embedding generation loops

### Our Enhanced RAG Solution:
✅ **Category-Specific Processing**: Research papers use heading-based chunking, receipts use line-item extraction, books preserve chapter structure

✅ **Intelligent Chunking**: Adaptive strategies based on document type maintain semantic coherence

✅ **Automatic Embeddings**: Langchain handles embedding generation automatically - no manual loops

✅ **Session Memory**: Maintains conversation history for multi-turn Q&A

✅ **Cloud-Based Storage**: ChromaDB Cloud integration with automatic cleanup

✅ **Production Ready**: Custom logging, exception handling, and error tracking

---

## 🏗️ Architecture Workflow

### High-Level Flow

```
┌─────────────┐
│  User       │  Uploads PDF → Asks Questions
└──────┬──────┘
       │
       ↓
┌──────────────────────────────────────┐
│  Streamlit UI (main.py)              │
│  - File upload & category detection  │
│  - Chat interface                    │
│  - Session management                │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│  RAG Agent (agent.py)                │
│  - Query processing                  │
│  - Context retrieval                 │
│  - Response generation               │
└──────┬───────────┬───────────────────┘
       │           │
       ↓           ↓
┌─────────────┐   ┌──────────────────┐
│ Vector Store│   │ Document         │
│ (Langchain  │   │ Processor        │
│  Chroma)    │   │ (PyPDF2)         │
└──────┬──────┘   └─────────┬────────┘
       │                    │
       ↓                    ↓
┌──────────────────────────────────────┐
│  AI Services (via Langchain)         │
│  - OpenAIEmbeddings                  │
│    (text-embedding-3-small)          │
│  - ChatOpenAI (gpt-4o-mini)          │
└──────┬───────────────────────────────┘
       │
       ↓
┌──────────────────────────────────────┐
│  External Services                   │
│  - AI/ML API (aimlapi.com)           │
│  - ChromaDB Cloud                    │
└──────────────────────────────────────┘
```

### Detailed Architecture

For detailed component interactions, data flow, and Langchain integration, see [Architecture Documentation](documentation/Main_documentation/architecture.md)

**Key Benefits of Our Stack:**
- 50% code reduction through Langchain automation
- Automatic embedding generation (no manual loops)
- Built-in retry logic and request batching
- Production-ready error handling

---

## 📁 File Structure

```
RAG-Enhanced-Chat-with-pdf/
│
├── main.py                          # Streamlit application entry point
├── pyproject.toml                   # uv dependency management
├── uv.lock                          # Dependency lock file
├── README.md                        # This file
├── .env                             # Environment variables (API keys)
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore rules
│
├── src/                             # Source code
│   ├── __init__.py                  # Package initialization
│   ├── agent.py                     # RAG agent with Langchain
│   ├── api_client.py                # Future API integrations
│   ├── config.py                    # Centralized configuration
│   ├── document_processor.py        # PDF processing & chunking
│   ├── embeddings.py                # Langchain embeddings wrapper
│   ├── exception.py                 # Custom exception classes
│   ├── llm_client.py                # Langchain ChatOpenAI wrapper
│   ├── logger.py                    # Custom logging system
│   ├── utils.py                     # Utility functions
│   ├── vector_store.py              # Langchain Chroma integration
│   └── logs/                        # Application logs (auto-created)
│
├── documentation/                   # Project documentation
│   ├── Main_documentation/          # Core documentation
│   │   ├── architecture.md          # System architecture details
│   │   ├── main_documentation.md    # Main project documentation
│   │   ├── the-art-of-chunking.md   # Chunking strategies
│   │   └── vector_search_and_storage.md  # Vector store details
│   ├── Artilce_and_Video/           # External resources
│   ├── Coding_Workflow/             # Development workflow docs
│   └── Future_Implementation/       # Future feature plans
│
├── tests/                           # Test suite
│
└── data/                            # Data storage (auto-created)
    ├── uploads/                     # Uploaded PDF files
    └── processed/                   # Processed document data
```

---

## 🛠️ How to Run This Setup

### Prerequisites

- **Python**: 3.10 or higher
- **uv**: Package manager (install from [astral.sh](https://docs.astral.sh/uv/))
- **ChromaDB Cloud Account**: Free tier available at [trychroma.com](https://www.trychroma.com/)
- **AI/ML API Key**: Get from [aimlapi.com](https://aimlapi.com/)

### 1. Clone the Repository

```bash
git clone https://github.com/nitanshuj/RAG-Enhanced-Chat-with-pdf.git
cd RAG-Enhanced-Chat-with-pdf
```

### 2. Install Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 3. ChromaDB Cloud Setup

#### Step 1: Create ChromaDB Cloud Account
1. Visit [https://www.trychroma.com/](https://www.trychroma.com/)
2. Sign up for a free account
3. Log in to your dashboard

#### Step 2: Create Database
1. Click **"Create Database"**
2. Name it `enhanced_rag_pdf_Q_and_A` (or your preferred name)
3. Copy the following credentials from your dashboard:
   - **API Key**
   - **Tenant ID**
   - **Database Name**

#### Step 3: Get AI/ML API Key
1. Visit [https://aimlapi.com/](https://aimlapi.com/)
2. Sign up and get your API key from the dashboard

#### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
# AI/ML API Configuration
AIMLAPI_KEY=your_aiml_api_key_here

# ChromaDB Cloud Configuration
CHROMA_API_KEY=your_chroma_api_key_here
CHROMA_TENANT=your_tenant_id_here
CHROMA_DATABASE=enhanced_rag_pdf_Q_and_A

# Optional: Disable telemetry
ANONYMIZED_TELEMETRY=False
```

**Security Note**: Never commit your `.env` file to version control. It's already in `.gitignore`.

#### Step 5: Verify Connection

Test your ChromaDB connection:

```bash
uv run python -c "from src.vector_store import VectorStore; vs = VectorStore(); print('✅ ChromaDB connection successful!')"
```

If successful, you'll see: `✅ ChromaDB connection successful!`

**Troubleshooting:**
- `CHROMA_API_KEY required` → Check your `.env` file
- `Connection timeout` → Verify internet connection and API key validity
- `Database not found` → Ensure database exists in ChromaDB dashboard

### 4. Run the Application

```bash
uv run streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

### 5. Using the Application

1. **Upload a PDF**: Click "Choose a PDF file" in the sidebar
2. **Confirm Category**: Auto-detection suggests a category, confirm or change it
3. **Process Document**: Click "🚀 Process Document"
4. **Ask Questions**: Type your questions in the chat input
5. **View Responses**: Get AI-generated answers based on your document
6. **Clear Session**: Click "🗑️ Clear" to delete document from cloud storage

---

## 🎯 Supported Document Categories

- **Research Paper**: Section-based chunking (Abstract, Methods, Results, etc.)
- **Article**: Paragraph-based chunking with topic preservation
- **Book**: Chapter-based hierarchical processing
- **Receipts**: Line-item extraction (experimental)
- **Terms & Conditions**: Clause-based processing
- **Other**: Adaptive chunking for general documents

---

## 🧪 Running Tests

```bash
# Run all tests
uv run python tests/run_tests.py

# Run specific test
uv run python tests/test_deletion.py
```

---

## 🔧 Configuration

Edit `src/config.py` to customize:

- **Model Selection**: Change LLM and embedding models
- **Chunk Sizes**: Adjust chunking parameters
- **RAG Parameters**: Modify top_k, context length, etc.
- **UI Settings**: Update app title, icon, layout

Example:

```python
# Models
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# RAG Parameters
DEFAULT_TOP_K = 5
MAX_CONTEXT_LENGTH = 2500
```

---

## 📊 Features

### ✅ Implemented
- Multi-category document processing
- Langchain-powered RAG pipeline
- ChromaDB Cloud vector storage
- Session-based conversation memory
- Automatic category detection
- Document deletion from cloud
- Custom logging system
- Exception handling
- Dark mode UI
- Confirmation dialogs

### 🚧 Future Enhancements
- Image and table extraction from PDFs
- Multiple file upload
- Export chat history
- Advanced RAG techniques (query expansion, re-ranking)
- Comprehensive test suite
- API documentation
- Caching and performance optimizations

---

## 📝 Logging

Logs are automatically saved to `src/logs/`:

- **Application logs**: `app_YYYYMMDD.log` (all events)
- **Error logs**: `error_YYYYMMDD.log` (errors only)

Log rotation: 10MB max size, 5 backup files

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🔗 Links

- **Repository**: [GitHub - RAG-Enhanced-Chat-with-pdf](https://github.com/nitanshuj/RAG-Enhanced-Chat-with-pdf)
- **ChromaDB**: [trychroma.com](https://www.trychroma.com/)
- **AI/ML API**: [aimlapi.com](https://aimlapi.com/)
- **Langchain**: [python.langchain.com](https://python.langchain.com/)
- **Streamlit**: [streamlit.io](https://streamlit.io/)

---

## 💡 Support

For issues, questions, or contributions:
- Open an issue on [GitHub Issues](https://github.com/nitanshuj/RAG-Enhanced-Chat-with-pdf/issues)
- Check [documentation](documentation/Main_documentation/) for detailed guides

---

**Built with ❤️ using Langchain, ChromaDB, and Streamlit**
