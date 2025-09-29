# Multi-Category RAG Chatbot - Coding Order

## Development Philosophy
- **Keep it simple**: Start with MVP and add complexity gradually
- **Test early**: Validate each component before moving to the next
- **Incremental approach**: Build working versions at each stage
- **Production-ready**: Focus on clean, maintainable code

---

## Phase 1: Foundation Setup (Days 1-2)

### 1.1 Environment Setup ✅ COMPLETED
- [x] ~~Create virtual environment~~ (Already done with uv) ✅
- [x] ~~Set up `.env` file for API keys~~ (Already configured with Chroma DB and AI/ML API keys) ✅
- [x] ~~Create basic `.gitignore`~~ (Already set up) ✅
- [x] Update dependencies in `pyproject.toml` for our specific needs: ✅
  ```toml
  dependencies = [
      "streamlit",
      "langchain",
      "langchain-community",
      "chromadb",              # For Chroma DB Cloud
      "openai",                # For AI/ML API compatibility
      "python-dotenv",
      "pypdf2",                # PDF processing only
      "pytesseract",           # OCR for receipts (PDF scanned documents)
      "pillow",                # Image processing (for PDF images)
      "requests",              # HTTP client for AI/ML API
      "numpy",                 # Array operations
      "pandas",                # Data manipulation
  ]
  ```
- [x] ~~Install dependencies: `uv sync`~~ ✅

### 1.2 Project Structure Creation ✅ COMPLETED
- [x] ~~Create `data/uploads/` directory~~ (Already created) ✅
- [x] ~~Create `data/processed/` directory~~ (Already created) ✅
- [x] ~~Create `data/vector_db/` directory~~ (Already created) ✅
- [x] ~~Initialize `src/__init__.py`~~ (Already exists) ✅

**Note**: Your project structure is perfectly set up! You have:
- ✅ `main.py` (Streamlit entry point)
- ✅ `pyproject.toml` (uv dependency management)
- ✅ `src/` with all required files (`agent.py`, `document_processor.py`, `llm_client.py`, etc.)
- ✅ `data/` directories (uploads, processed, vector_db)
- ✅ `.env` with API keys configured
- ✅ `.gitignore` properly set up
- ✅ Documentation structure in place

### 1.3 Basic Streamlit App (`main.py`) ✅ COMPLETED

Create minimal Streamlit interface with: 
- File upload widget                ✅
- Document category selector (with all categories defined in the main documentation - "main_documentation.md") ✅
- Basic chat interface              ✅
- Session state management          ✅


**Goal**: Working file upload interface that accepts PDFs

---

## Phase 2: Document Processing Core (Days 3-4)

### 2.1 Basic Document Processor (`src/document_processor.py`)
```python
class DocumentProcessor:
    def __init__(self):
        # Initialize PDF parser only

    def extract_text(self, file_buffer):
        # PDF text extraction only

    def simple_chunk(self, text, chunk_size=500):
        # Simple text chunking

    def process_document(self, file_buffer, category):
        # Main processing pipeline for PDF only
```

**Features to implement**:
- [x] ~~PDF text extraction (PyPDF2)~~ ✅
- [x] ~~Basic chunking (fixed size with overlap)~~ ✅
- [x] ~~Category-aware metadata extraction~~ ✅
- [x] ~~Return processed chunks (no disk storage needed)~~ ✅

**Goal**: Convert uploaded documents to text chunks

### 2.2 AI/ML API Client (`src/llm_client.py`)
```python
class LLMClient:
    def __init__(self, api_key, base_url="https://api.aimlapi.com"):
        # Initialize API client

    def generate_embeddings(self, text):
        # Generate embeddings via API

    def generate_response(self, prompt, context):
        # Generate chat responses

    def classify_document(self, text_sample):
        # Auto-classify document category
```

**Features to implement**:
- [x] ~~API connection to aimlapi.com~~ ✅
- [x] ~~Text embedding generation~~ ✅
- [x] ~~Chat completion generation~~ ✅
- [x] ~~Error handling and retries~~ ✅
- [x] ~~Rate limiting~~ ✅

**Goal**: Working API integration for embeddings and chat

---

## Phase 3: Vector Store Integration (Days 5-6)

### 3.1 Vector Store Operations (`src/vector_store.py`)
```python
class VectorStore:
    def __init__(self, persist_directory="data/vector_db"):
        # Initialize Chroma client

    def add_documents(self, chunks, metadata, embeddings):
        # Store document chunks with metadata

    def similarity_search(self, query, category_filter=None, top_k=5):
        # Retrieve relevant chunks

    def delete_document(self, document_id):
        # Remove document from store
```

**Features to implement**:
- [ ] Chroma database initialization
- [ ] Document storage with category metadata
- [ ] Similarity search with filtering
- [ ] Persistence to disk
- [ ] Basic metadata structure

**Goal**: Store and retrieve document chunks effectively

### 3.2 Integration Testing
- [ ] Test document upload � processing � storage pipeline
- [ ] Verify embeddings are generated correctly
- [ ] Test retrieval with different categories
- [ ] Validate metadata filtering works

---

## Phase 4: Basic RAG Implementation (Days 7-8)

### 4.1 Simple RAG Agent (`src/agent.py`)
```python
class SimpleRAGAgent:
    def __init__(self, llm_client, vector_store):
        # Initialize components

    def process_query(self, query, category_filter=None):
        # Basic RAG: retrieve � generate

    def format_context(self, retrieved_chunks):
        # Format context for LLM

    def generate_answer(self, query, context):
        # Generate final response
```

**Features to implement**:
- [ ] Query processing pipeline
- [ ] Context retrieval and formatting
- [ ] Response generation
- [ ] Basic prompt engineering
- [ ] Simple session state

**Goal**: Working Q&A system for uploaded documents

### 4.2 Streamlit Integration ✅ COMPLETED
- [x] ~~Connect agent to Streamlit interface~~ ✅
- [x] ~~Display retrieved context (optional debug view)~~ ✅
- [x] ~~Show response with sources~~ ✅
- [x] ~~Add basic error handling in UI~~ ✅

---

## Phase 5: Category-Specific Processing (Days 9-11)

### 5.1 Enhanced Document Processor
```python
class CategoryProcessor:
    def get_category_chunking_strategy(self, category):
        # Different chunking per category

    def extract_category_metadata(self, text, category):
        # Category-specific metadata extraction

    def preprocess_by_category(self, text, category):
        # Category-specific preprocessing
```

**Category implementations**:
- [x] ~~**Research Paper**: Section-based chunking (Abstract, Methods, Results, etc.)~~ ✅
- [x] ~~**Article**: Paragraph-based with topic extraction~~ ✅
- [x] ~~**Book**: Chapter-based hierarchy~~ ✅
- [x] ~~**Receipts**: Line-item extraction with OCR~~ ✅
- [x] ~~**Terms & Conditions**: Clause-based processing~~ ✅
- [x] ~~**Other**: Adaptive chunking~~ ✅

**Goal**: Specialized processing for each document type

### 5.2 Enhanced Retrieval ✅ COMPLETED
- [x] ~~Category-specific prompt templates~~ ✅
- [x] ~~Metadata-based filtering improvements~~ ✅
- [x] ~~Category-aware response formatting~~ ✅

---

## Phase 6: Agentic RAG Features (Days 12-14)

### 6.1 Multi-Step Query Processing
```python
class AgenticRAG:
    def decompose_query(self, query):
        # Break complex queries into subtasks

    def execute_subtask(self, subtask, context):
        # Process individual subtasks

    def synthesize_response(self, subtask_results):
        # Combine results into final answer
```

**Features to implement**:
- [ ] Query decomposition logic
- [ ] Multi-step retrieval
- [ ] Context chaining between steps
- [ ] Response synthesis

### 6.2 Session Memory
```python
class SessionMemory:
    def store_interaction(self, query, response, context):
        # Store chat history

    def get_relevant_history(self, current_query):
        # Retrieve relevant past interactions

    def update_session_context(self, new_insights):
        # Maintain session-level insights
```

**Goal**: Conversational context and multi-turn reasoning

---

## Phase 7: Production Enhancements (Days 15-17)

### 7.1 Error Handling & Validation
- [ ] Input validation for all file types
- [ ] API error handling with fallbacks
- [ ] User-friendly error messages
- [ ] File size and type limitations
- [ ] Graceful degradation

### 7.2 Performance Optimization
- [ ] Caching for repeated queries
- [ ] Async processing where possible
- [ ] Progress indicators for long operations
- [ ] Memory management for large documents

### 7.3 User Experience
- [ ] Document management interface
- [ ] Category auto-detection
- [ ] Response quality indicators
- [ ] Export/save functionality

---

## Phase 8: Advanced Features (Days 18-20)

### 8.1 Enhanced Document Support
- [ ] Image extraction from PDFs
- [ ] Table processing
- [ ] Multiple file upload
- [ ] Document comparison features

### 8.2 Advanced RAG Features
- [ ] Confidence scoring
- [ ] Source attribution
- [ ] Query expansion
- [ ] Response validation

---

## Testing Strategy

### Unit Tests (Throughout development)
- [ ] Document processing functions
- [ ] Vector store operations
- [ ] LLM client methods
- [ ] Utility functions

### Integration Tests (End of each phase)
- [ ] End-to-end document processing
- [ ] RAG pipeline testing
- [ ] Category-specific workflows
- [ ] Error scenarios

### User Acceptance Testing (Phase 7-8)
- [ ] Test with real documents
- [ ] Validate category processing
- [ ] Performance testing
- [ ] UI/UX validation

---

## Deployment Preparation

### Configuration Management
- [ ] Environment-specific configs
- [ ] API key management
- [ ] Database connection settings
- [ ] Logging configuration

### Documentation
- [ ] API documentation
- [ ] User manual
- [ ] Deployment guide
- [ ] Troubleshooting guide

### Security
- [ ] Input sanitization
- [ ] File upload security
- [ ] API key protection
- [ ] Data privacy compliance

---

## Development Guidelines

1. **Dependency Management**:
   - Use `uv add <package>` to add new dependencies
   - Use `uv sync` to install/update dependencies
   - Use `uv run <command>` to run scripts in the virtual environment
   - Example: `uv run streamlit run main.py`

2. **Code Quality**:
   - Use type hints
   - Write docstrings
   - Follow PEP 8
   - Keep functions small and focused

3. **Git Workflow**:
   - Commit after each major feature
   - Use descriptive commit messages
   - Create feature branches for major changes

4. **Documentation**:
   - Update README as you progress
   - Document API changes
   - Keep inline comments minimal but meaningful

5. **Testing**:
   - Test each component as you build it
   - Use small test documents initially
   - Validate with real documents before moving to next phase
   - Run tests with: `uv run python -m pytest tests/`

---

## Success Criteria

- **Phase 1-3**: Upload and process any document type
- **Phase 4**: Answer basic questions about uploaded documents
- **Phase 5**: Demonstrate category-specific processing differences
- **Phase 6**: Handle complex, multi-part questions
- **Phase 7**: Production-ready with proper error handling
- **Phase 8**: Advanced features working smoothly

---

## Estimated Timeline: 20 development days

**Note**: This timeline assumes dedicated development time. Adjust based on your availability and complexity requirements.