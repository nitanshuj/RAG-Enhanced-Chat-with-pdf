# RAG-Enhanced Chat with PDF - System Architecture (Langchain-Based)

This document describes the complete system architecture using **Langchain framework**, including the flow of data from user input to LLM response generation.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Layer-by-Layer Description](#layer-by-layer-description)
3. [Data Flow Diagram](#data-flow-diagram)
4. [Component Interactions](#component-interactions)
5. [Configuration Management](#configuration-management)
6. [Request Flow Examples](#request-flow-examples)

---

## Architecture Overview

The system follows a **layered architecture with Langchain framework** for simplified RAG operations:

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 5: USER INTERFACE                  │
│                         (main.py)                           │
│  - Streamlit UI                                             │
│  - File upload handling                                     │
│  - Chat interface                                           │
│  - Session management                                       │
└────────────────────────┬────────────────────────────────────┘
                         │ calls
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              LAYER 4: RAG ORCHESTRATION (LANGCHAIN)         │
│                        (agent.py)                           │
│  - SimpleRAGAgent: Complete RAG pipeline                    │
│  - Hybrid search (BM25 + Dense vectors)                     │
│  - Cross-encoder re-ranking                                 │
│  - Response caching (cache.py)                              │
│  - Semantic memory (VectorStoreRetrieverMemory)             │
│  - Web search integration (web_search.py)                   │
│  - Streaming responses                                      │
│  - Langchain PromptTemplate                                 │
│  - Query processing & validation                            │
│  - Conversation history management                          │
│  - Context formatting                                       │
└──────────┬──────────────────────────────────┬───────────────┘
           │ uses                             │ uses
           ↓                                  ↓
┌─────────────────────────────────────────────────────────────┐
│              LAYER 3: DATA & UTILITIES                      │
├──────────────────────────┬──────────────────────────────────┤
│   vector_store.py        │    document_processor.py         │
│  - Langchain Chroma      │    - PDF text extraction         │
│  - BM25 + Dense search   │    - Category detection          │
│  - Ensemble retriever    │    - Semantic chunking           │
│  - Cross-encoder rerank  │    - Rule-based chunking         │
│  - Auto embeddings       │    - Metadata extraction         │
│  - Metadata filtering    │                                  │
├──────────────────────────┼──────────────────────────────────┤
│   cache.py               │    web_search.py                 │
│  - ResponseCache         │    - WebSearcher                 │
│  - In-memory TTL cache   │    - DuckDuckGo integration      │
│  - Hash-based keys       │    - Result formatting           │
│  - Cache statistics      │    - Hybrid doc+web results      │
│  - 30-50% cost reduction │    - 85% docs + 15% web          │
└────────────┬─────────────┴──────────────┬───────────────────┘
             │ uses Langchain             │ uses
             ↓                            ↓
┌─────────────────────────────────────────────────────────────┐
│              LAYER 2: LANGCHAIN AI SERVICES                 │
├──────────────────────────┬──────────────────────────────────┤
│   embeddings.py          │       llm_client.py              │
│                          │                                  │
│  - OpenAIEmbeddings      │  - ChatOpenAI                    │
│  - embed_query()         │  - invoke() with messages        │
│  - embed_documents()     │  - SystemMessage/HumanMessage    │
└────────────┬─────────────┴──────────────┬───────────────────┘
             │ uses Langchain             │ uses Langchain
             └────────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│            LAYER 1: LANGCHAIN BASE CONFIG                   │
│                   (config.py)                               │
│                                                             │
│  - AI/ML API configuration                                 │
│  - Model names (gpt-4o-mini, text-embedding-3-small)       │
│  - API base URL & keys                                     │
│  - RAG parameters                                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   AI/ML API BACKEND                         │
│            (https://api.aimlapi.com/v1)                     │
│                                                             │
│  - GPT-4o-mini (Chat Completions)                          │
│  - text-embedding-3-small (Embeddings)                     │
│  - OpenAI-compatible API                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer-by-Layer Description

### **Layer 1: Configuration (`src/config.py`)**

**Purpose:** Centralize all configuration for Langchain components.

**Key Components:**
```python
# Models
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# AI/ML API Settings
AIML_API_BASE_URL = "https://api.aimlapi.com/v1"
AIML_API_KEY = os.getenv("AIMLAPI_KEY")

# RAG Parameters
DEFAULT_TOP_K = 5
MAX_CONTEXT_LENGTH = 2500
```

**Why it exists:**
- Single source of truth for Langchain initialization
- Easy model switching without code changes
- Environment-specific configuration

---

### **Layer 2: Langchain AI Services**

#### **`src/embeddings.py` - Langchain OpenAI Embeddings**

**Purpose:** Generate embeddings using Langchain's OpenAIEmbeddings.

**Key Components:**
```python
from langchain_openai import OpenAIEmbeddings

def get_embeddings_model():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_base=AIML_API_BASE_URL,
        openai_api_key=AIML_API_KEY
    )

def generate_embeddings(text: str):
    embeddings_model = get_embeddings_model()
    embedding = embeddings_model.embed_query(text)
    return True, embedding, None
```

**Benefits:**
- **Automatic batching** - Langchain handles batch optimization
- **Built-in retry logic** - Langchain manages API failures
- **Simplified API** - Single line: `embed_query(text)`
- **No manual HTTP** - Langchain handles all requests

**Before (Custom):** ~240 lines with manual API calls
**After (Langchain):** ~140 lines with automatic handling

---

#### **`src/llm_client.py` - Langchain ChatOpenAI**

**Purpose:** Generate responses using Langchain's ChatOpenAI.

**Key Components:**
```python
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

def get_llm_model(temperature=None, max_tokens=None):
    return ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_base=AIML_API_BASE_URL,
        openai_api_key=AIML_API_KEY,
        temperature=temperature or DEFAULT_CHAT_TEMPERATURE,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS
    )

def generate_response(prompt: str, context: str = None):
    llm = get_llm_model()
    messages = [
        SystemMessage(content="You are an expert..."),
        HumanMessage(content=f"Context: {context}\n\nQuestion: {prompt}")
    ]
    response = llm.invoke(messages)
    return True, response.content, None
```

**Benefits:**
- **Message abstractions** - SystemMessage, HumanMessage types
- **Streaming support** - Easy to add streaming responses
- **Chain compatibility** - Works with Langchain chains
- **No manual HTTP** - Langchain handles requests

**Before (Custom):** ~250 lines with manual message building
**After (Langchain):** ~200 lines with message objects

---

### **Layer 3: Data & Document Processing**

#### **`src/vector_store.py` - Langchain Chroma**

**Purpose:** Manage vector storage using Langchain's Chroma integration.

**Key Components:**
```python
from langchain_chroma import Chroma
from langchain.schema import Document
import chromadb

class VectorStore:
    def __init__(self, collection_name="documents"):
        # Initialize ChromaDB Cloud client
        client = chromadb.CloudClient(
            tenant=self.tenant,
            database=self.database,
            api_key=self.api_key
        )

        # Initialize Langchain Chroma (handles embeddings automatically)
        self.vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=get_embeddings_model()
        )

    def add_documents(self, chunks, metadata, embeddings=None):
        # Create Langchain Documents
        documents = [
            Document(page_content=chunk, metadata=meta)
            for chunk, meta in zip(chunks, metadata)
        ]

        # Add documents (Langchain handles embedding automatically!)
        ids = self.vectorstore.add_documents(documents)
        return True, f"Added {len(documents)} chunks", ids

    def similarity_search(self, query_text, category_filter=None, top_k=5):
        # Langchain handles embedding query automatically
        docs = self.vectorstore.similarity_search(
            query=query_text,
            k=top_k,
            filter={"category": category_filter} if category_filter else None
        )
        return True, docs, None
```

**Benefits:**
- **Automatic embeddings** - No manual embedding generation needed
- **Document abstraction** - Langchain Document type
- **Simplified search** - Pass text directly, no need to embed first
- **Metadata filtering** - Built-in filter support

**Before (Custom):** ~450 lines, manual embedding handling
**After (Langchain):** ~220 lines, automatic embeddings

**Critical Change:** Main.py no longer needs embedding loops!

**Before:**
```python
# Manual embedding loop (REMOVED)
embedding_list = []
for chunk in chunks:
    success, embedding, error = embeddings.generate_embeddings(chunk)
    embedding_list.append(embedding)

store.add_documents(chunks, metadata, embedding_list)
```

**After:**
```python
# Langchain handles embeddings automatically
store.add_documents(chunks, metadata, None)  # embeddings=None!
```

---

#### **`src/document_processor.py`**

**Purpose:** Process PDF documents (unchanged, not Langchain-specific).

**Key Functions:**
- `extract_text()` - PDF text extraction (PyPDF2)
- `heading_based_chunk()` - Research paper chunking
- `simple_chunk()` - Standard chunking

**No changes needed** - Document processing is independent of Langchain.

---

### **Layer 4: RAG Orchestration (`src/agent.py`)**

**Purpose:** Orchestrate RAG pipeline using Langchain components.

**Key Components:**
```python
from langchain.prompts import PromptTemplate
from src.llm_client import get_llm_model

class SimpleRAGAgent:
    def __init__(self, llm_client, vector_store):
        self.vector_store = vector_store
        self.llm = get_llm_model()

        # Langchain PromptTemplate
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert document analysis assistant.

Context:
{context}

Question: {question}

Answer:"""
        )

    def process_query(self, query, category_filter=None, top_k=5):
        # Step 1: Retrieve (Langchain handles embedding automatically)
        success, retrieved_docs, error = self.vector_store.similarity_search(
            query_text=query,  # Pass text directly!
            category_filter=category_filter,
            top_k=top_k
        )

        # Step 2: Format context
        context = self._format_context(retrieved_docs)

        # Step 3: Generate answer with Langchain
        prompt = self.prompt_template.format(
            context=context,
            question=query
        )

        from langchain.schema import HumanMessage
        response = self.llm.invoke([HumanMessage(content=prompt)])

        return True, {"answer": response.content, "sources": retrieved_docs}, None
```

**Benefits:**
- **PromptTemplate** - Reusable prompt structure
- **Direct text search** - No manual embedding step
- **Chain ready** - Can upgrade to RetrievalQA chain
- **Simplified flow** - Fewer steps, cleaner code

**Before (Custom):**
1. Validate query
2. Generate query embedding manually
3. Search with embedding
4. Format context
5. Generate answer

**After (Langchain):**
1. Validate query
2. Search with text (Langchain embeds automatically)
3. Format context
4. Generate answer

**Removed step:** Manual query embedding generation!

---

### **Layer 5: User Interface (`main.py`)**

**Purpose:** Streamlit web interface.

**Key Changes with Langchain:**

**Before:**
```python
# Manual embedding loop (116-130 lines)
embedding_list = []
embedding_progress = st.progress(0)

for i, chunk in enumerate(chunks):
    success_emb, embedding, error = embeddings.generate_embeddings(chunk)
    if success_emb:
        embedding_list.append(embedding)
    else:
        embedding_list.append(None)
        st.warning(f"Failed to generate embedding for chunk {i+1}")

    embedding_progress.progress((i + 1) / len(chunks))

embedding_progress.empty()

store_success, store_message, chunk_ids = vector_store.add_documents(
    chunks, vector_metadata, embedding_list
)
```

**After:**
```python
# Langchain handles embeddings automatically (117-120 lines)
with st.spinner("Generating embeddings and storing documents..."):
    store_success, store_message, chunk_ids = vector_store.add_documents(
        chunks, vector_metadata, None  # No embeddings needed!
    )
```

**Benefits:**
- **15 lines removed** - No embedding loop
- **Cleaner UI** - Single progress spinner
- **Faster** - Langchain batches embeddings efficiently

---

## Data Flow Diagram

### **Document Upload & Processing Flow (Langchain)**

```
┌──────────────┐
│  User        │
│  Uploads PDF │
└──────┬───────┘
       │
       ↓
┌──────────────────────────┐
│ main.py                  │
│ - auto_detect_category() │
│ - User confirms          │
└──────┬───────────────────┘
       │
       ↓
┌────────────────────────────────┐
│ document_processor.py          │
│ - extract_text()               │
│ - heading_based_chunk()        │
│ - extract_metadata()           │
└──────┬─────────────────────────┘
       │ Returns: chunks + metadata
       ↓
┌────────────────────────────────┐
│ vector_store.py (Langchain)    │
│ - add_documents(chunks, meta)  │
│ - Langchain AUTOMATICALLY:     │
│   • Generates embeddings       │
│   • Batches requests           │
│   • Stores in ChromaDB         │
└──────┬─────────────────────────┘
       │ No manual embedding loop!
       ↓
┌──────────────────────────┐
│ Document Ready for Q&A   │
└──────────────────────────┘
```

**Key Difference:** No manual embedding loop in main.py!

---

### **Question Answering Flow (Langchain)**

```
┌──────────────┐
│  User        │
│  Asks Query  │
└──────┬───────┘
       │
       ↓
┌──────────────────────────────────────┐
│ main.py                              │
│ - agent.process_query(user_question) │
└──────┬───────────────────────────────┘
       │
       ↓
┌────────────────────────────────────────────┐
│ agent.py - SimpleRAGAgent (Langchain)      │
│ ┌────────────────────────────────────────┐ │
│ │ Step 1: Validate Query                 │ │
│ └────────────────────────────────────────┘ │
│                  ↓                          │
│ ┌────────────────────────────────────────┐ │
│ │ Step 2: Retrieve Documents             │ │
│ │ - vector_store.similarity_search()     │ │
│ │ - Pass QUERY TEXT directly             │ │────→ Langchain Chroma
│ │ - Langchain embeds query automatically │ │       ↓
│ │ - Returns top_k chunks                 │ │   OpenAIEmbeddings
│ └────────────────────────────────────────┘ │       ↓
│                  ↓                          │   ChromaDB Cloud
│ ┌────────────────────────────────────────┐ │
│ │ Step 3: Format Context                 │ │
│ │ - PromptTemplate.format()              │ │
│ │ - Combine chunks with metadata         │ │
│ └────────────────────────────────────────┘ │
│                  ↓                          │
│ ┌────────────────────────────────────────┐ │
│ │ Step 4: Generate Answer                │ │
│ │ - ChatOpenAI.invoke(messages)          │ │────→ Langchain ChatOpenAI
│ │ - SystemMessage + HumanMessage         │ │       ↓
│ └────────────────────────────────────────┘ │   AI/ML API
│                  ↓                          │   (gpt-4o-mini)
│ ┌────────────────────────────────────────┐ │
│ │ Step 5: Update Session                 │ │
│ │ - Store in conversation_history        │ │
│ └────────────────────────────────────────┘ │
└──────┬─────────────────────────────────────┘
       │ Returns: (success, response_data, error)
       ↓
┌──────────────────────────┐
│ main.py                  │
│ - Display answer to user │
└──────────────────────────┘
```

**Key Difference:** No manual query embedding generation!

---

## Component Interactions (Langchain Stack)

### **Dependency Graph**

```
main.py
  ├── imports: agent, document_processor, vector_store
  └── calls: agent.process_query()

agent.py (Langchain)
  ├── imports: langchain.prompts.PromptTemplate
  ├── imports: llm_client.get_llm_model (ChatOpenAI)
  ├── imports: cache.get_cache (ResponseCache)
  ├── imports: web_search.get_web_searcher (WebSearcher)
  ├── calls: vector_store.hybrid_search() [BM25 + Dense]
  ├── calls: vector_store.rerank_results() [Cross-encoder]
  ├── calls: web_searcher.search() [DuckDuckGo]
  └── calls: llm.invoke() [Langchain ChatOpenAI]

vector_store.py (Langchain)
  ├── imports: langchain_chroma.Chroma
  ├── imports: langchain.schema.Document
  ├── imports: langchain_community.retrievers.BM25Retriever
  ├── imports: langchain.retrievers.EnsembleRetriever
  ├── imports: sentence_transformers.CrossEncoder
  ├── imports: embeddings.get_embeddings_model()
  └── uses: Langchain auto-embedding on add/search

cache.py
  ├── implements: ResponseCache (in-memory TTL cache)
  ├── methods: get(), set(), clear(), get_stats()
  └── features: Hash-based keys, TTL expiration, cache statistics

web_search.py
  ├── imports: duckduckgo_search.DDGS
  ├── implements: WebSearcher (web search integration)
  ├── methods: search(), should_use_web_search(), combine_results()
  └── features: DuckDuckGo search, result formatting, hybrid results

embeddings.py (Langchain)
  ├── imports: langchain_openai.OpenAIEmbeddings
  └── returns: OpenAIEmbeddings instance

llm_client.py (Langchain)
  ├── imports: langchain_openai.ChatOpenAI
  ├── imports: langchain.schema (SystemMessage, HumanMessage)
  └── returns: ChatOpenAI instance

config.py
  └── provides: API keys, model names, base URLs
```

---

## Configuration Management

### **Langchain Model Configuration**

Edit `src/config.py`:

```python
# Chat Model (Langchain ChatOpenAI)
CHAT_MODEL = "gpt-4o-mini"  # Used by ChatOpenAI

# Embedding Model (Langchain OpenAIEmbeddings)
EMBEDDING_MODEL = "text-embedding-3-small"  # Used by OpenAIEmbeddings

# AI/ML API Configuration (OpenAI-compatible)
AIML_API_BASE_URL = "https://api.aimlapi.com/v1"
AIML_API_KEY = os.getenv("AIMLAPI_KEY")

# Langchain Parameters
DEFAULT_CHAT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1000
```

**Langchain automatically uses these settings when initialized:**

```python
# Embeddings automatically use EMBEDDING_MODEL
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_base=AIML_API_BASE_URL,
    openai_api_key=AIML_API_KEY
)

# Chat automatically uses CHAT_MODEL
llm = ChatOpenAI(
    model=CHAT_MODEL,
    openai_api_base=AIML_API_BASE_URL,
    openai_api_key=AIML_API_KEY
)
```

---

## Request Flow Examples

### **Example: Simple Question (Langchain)**

**User:** "What is this document about?"

**Flow:**
1. `main.py` → `agent.process_query("What is this document about?")`
2. `agent.py` → `vector_store.similarity_search(query_text="What is this...")`
   - **Langchain Chroma automatically embeds query internally**
3. ChromaDB returns 5 most similar chunks
4. `agent.py` formats context using PromptTemplate
5. `agent.py` → `llm.invoke(messages)` [Langchain ChatOpenAI]
   - **Langchain handles HTTP request to AI/ML API**
6. API returns answer
7. `agent.py` stores in conversation history
8. `main.py` displays answer

**Total manual steps:** 4 (was 6 before Langchain)
**Removed steps:**
- Manual query embedding generation
- Manual HTTP request handling

---

## Key Benefits of Langchain Architecture

### **Code Reduction**
- **embeddings.py:** 240 → 140 lines (42% reduction)
- **llm_client.py:** 250 → 200 lines (20% reduction)
- **vector_store.py:** 450 → 220 lines (51% reduction)
- **agent.py:** 415 → 210 lines (49% reduction)
- **main.py:** Removed 15-line embedding loop

**Total:** ~50% code reduction

### **Simplified Operations**
```python
# Before (Custom):
embedding = generate_embeddings(text)
results = store.search(embedding)

# After (Langchain):
results = store.search(text)  # Langchain handles embedding!
```

### **Production Features (Free)**
- **Automatic retry logic** - Langchain handles API failures
- **Request batching** - Optimized embedding generation
- **Streaming support** - Easy to add streaming responses
- **Chain compatibility** - Can use RetrievalQA, ConversationalChain
- **Memory integration** - ConversationBufferMemory ready

### **Extensibility**
Easy to add Langchain features:
- **Chains:** RetrievalQA, ConversationalRetrievalChain
- **Agents:** React agents with tools
- **Memory:** Multiple memory types (buffer, summary, etc.)
- **Callbacks:** Logging, monitoring, debugging

---

## Summary

**Langchain Architecture Provides:**

✅ **50% less code** - Automatic embedding and HTTP handling
✅ **Simpler operations** - Search with text, not vectors
✅ **Production-ready** - Built-in retry, batching, error handling
✅ **Extensible** - Easy to add chains, agents, memory
✅ **Maintainable** - Standard Langchain patterns
✅ **Modern stack** - Using gpt-4o-mini via AI/ML API

**The golden rule:** Langchain handles low-level operations (embedding, HTTP), we focus on business logic (chunking, RAG orchestration).
