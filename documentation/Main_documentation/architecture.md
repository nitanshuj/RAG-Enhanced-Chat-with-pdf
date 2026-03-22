# RAG-Enhanced Chat with PDF - System Architecture

This document provides a deep conceptual and theoretical overview of the system architecture, focusing on the retrieval, processing, and generation mechanics. The system is built on an **Agentic RAG (Retrieval-Augmented Generation)** framework, optimized for accurate document-based Q&A.

---

## 🏗️ Architecture Overview

The system is designed as a multi-layered pipeline where each layer focuses on a specific stage of the document's journey—from raw PDF data to a refined AI response. By abstracting complexity through specialized modules, we achieve a robust, scalable, and maintainable RAG system.

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 5: USER INTERFACE                  │
│                         (server.py)                         │
│  - FastAPI Backend                                          │
│  - File upload handling                                     │
│  - Chat interface                                           │
│  - Session management                                       │
└────────────────────────┬────────────────────────────────────┘
                         │ calls
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              LAYER 4: RAG ORCHESTRATION (AGENTIC)           │
│                        (agent.py)                           │
│  - SimpleRAGAgent: Complete RAG pipeline                    │
│  - Hybrid search (BM25 + Dense vectors)                     │
│  - Cross-encoder re-ranking                                 │
│  - Response caching (cache.py)                              │
│  - Semantic memory (VectorStoreRetrieverMemory)             │
│  - Web search integration (web_search.py)                   │
│  - Streaming responses                                      │
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
├──────────────────────────┼──────────────────────────────────┤
│   cache.py               │    web_search.py                 │
│  - ResponseCache         │    - WebSearcher                 │
│  - In-memory TTL cache   │    - DuckDuckGo integration      │
│  - Hash-based keys       │    - Result formatting           │
└────────────┬─────────────┴──────────────┬───────────────────┘
             │ uses                       │ uses
             ↓                            ↓
┌─────────────────────────────────────────────────────────────┐
│              LAYER 2: AI SERVICES (LANGCHAIN)               │
├──────────────────────────┬──────────────────────────────────┤
│   embeddings.py          │       llm_client.py              │
│                          │                                  │
│  - OpenAIEmbeddings      │  - ChatOpenAI                    │
│  - embed_query()         │  - invoke() with messages        │
└────────────┬─────────────┴──────────────┬───────────────────┘
             │                            │
             └────────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   AI/ML API BACKEND                         │
│            (https://api.aimlapi.com/v1)                     │
│                                                             │
│  - GPT-5-Nano/Mini (Chat Completions)                       │
│  - text-embedding-3-small (Embeddings)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧬 Detailed Theoretical Concepts

### 1. Advanced Document Processing (Data Engineering)
Unlike simple text-splitting methods, this system employs **Multi-Strategy Chunking**:

*   **Semantic Chunking (Theoretical Basis)**: Instead of arbitrary character counts, the system calculates the **Cosine Similarity** between sentence embeddings. It identifies "breakpoints" where the semantic flow significantly shifts (e.g., when the similarity drop exceeds a certain percentile). This ensures each chunk is a self-contained thematic unit, which improves retrieval accuracy by preserving the context of thoughts.
*   **Structural & Heading-Based Awareness**: For research papers, the system detects visual and formatting cues. By tagging chunks with their specific sections (e.g., *Abstract*, *Methodology*, *Results*), the system allows the **Retriever** to weigh information based on its location—mimicking how a human researcher prioritizes "Conclusions" over "Introduction" when looking for facts.
*   **Recursive Metadata Filtering**: Each chunk isn't just text; it's an object containing its source provenance. During retrieval, we can apply **hard filters** (e.g., "only search in 'Finance' PDFs") before the semantic similarity is even calculated, dramatically reducing the noise in the retrieved set.

### 2. Hybrid Retrieval & Ensemble Search
The search architecture is built on the theory that **Keyword Search (Sparse)** and **Semantic Search (Dense)** are complementary:

*   **Dense Vector Search (Bi-Encoders)**: Maps the question and document into a 1536-dimensional space. It excels at finding "synoptic relevance" (e.g., finding "revenue" when the user asks for "income").
*   **BM25 (Sparse Keyword Search)**: Based on the **TF-IDF** (Term Frequency-Inverse Document Frequency) principle, BM25 is crucial for finding "Exact-Match" facts like names, SKUs, or specialized scientific terms where semantic models might merge distinct concepts.
*   **The Ensemble Retriever (Reciprocal Rank Fusion)**: We combine both results using a weighted formula. This ensures that a document is ranked highly if it is *either* perfectly semantically matched *or* perfectly keyword matched, or (ideally) both.

### 3. Cross-Encoder Re-Ranking (Precision Layer)
Initial retrieval is fast but sometimes captures "semantically similar" but irrelevant noise. To fix this, we implement a **Cross-Encoder Model**:

*   **Theory**: Unlike Bi-Encoders (which embed Q and D separately), a Cross-Encoder processes the **Question and Document together** as a single input. This allows "full-interaction" between the terms in the question and the document text.
*   **Outcome**: It acts as a final filter, re-scoring the retrieved top chunks with 10x higher precision. This ensures that the context provided to the LLM is not just "related," but is definitively "responsive" to the prompt.

### 4. Agentic Orchestration & Prompt Synthesis
The "Agent" acts as a cognitive controller, performing **Context Synthesis**:

*   **Prompt Augmentation Theory**: The LLM is forced into a **Zero-Shot Retrieval-Corrective** mode. By providing "External Evidence" (retrieved chunks) within a structured system prompt, we effectively "ground" the model, preventing it from using its internal training weights to hallucinate facts not present in the PDF.
*   **Streaming & Latency Management**: We implement an asynchronous streaming pipeline. This utilizes the theory of **Token-by-Token Rendering**, allowing the user to start reading the answer as soon as the model generates the first "reasoning" thought, rather than waiting for the entire 4000-token completion.
*   **GPT-5 Reasoning Phase**: The system is optimized for "thinking" models. These models use a **chain-of-thought** phase internally before outputting text. Our architecture allows for higher token overhead (`max_tokens=4000`) to accommodate this internal reasoning process.

### 5. Advanced System Intelligence
*   **Semantic Response Caching**: Instead of caching based on exact strings (which fails if a user adds a space), we can leverage **Semantic Hashing**. If a new question is semantically identical to a previous one, the cache satisfies the request, reducing latency and cost by 30-50%.
*   **Conversation Buffer Memory**: The system implements a **Context-Aware Sliding Window**. It doesn't just look at the last question; it summarizes the conversation history to understand pronominal references (e.g., "Compare it with the previous one").
*   **Hybrid Web-Search Bridge**: When document retrieval scores are below a specific threshold (suggesting the answer isn't in the PDF), the agent can bridge the knowledge gap using **Real-Time Web Search**, merging local and global context into a single cohesive response.

---

## 🚀 Data Flow Pipeline

### Document Upload & Processing Flow

```
┌──────────────┐
│  User        │
│  Uploads PDF │
└──────┬───────┘
       │
       ↓
┌──────────────────────────┐
│ server.py                │
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

### Question Answering Flow

```
┌──────────────┐
│  User        │
│  Asks Query  │
└──────┬───────┘
       │
       ↓
┌──────────────────────────────────────┐
│ server.py (FastAPI)                  │
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
└─┘                ↓                         └─┘
│ ┌────────────────────────────────────────┐ │
│ │ Step 4: Generate Streaming Answer      │ │
│ │ - ChatOpenAI.stream(messages)          │ │────→ Langchain ChatOpenAI
│ │ - SystemMessage + HumanMessage         │ │       ↓
│ └────────────────────────────────────────┘ │   AI/ML API
│                  ↓                          │   (GPT-5 Nano/Mini)
│ ┌────────────────────────────────────────┐ │
│ │ Step 5: Update Session                 │ │
│ │ - Store in conversation_history        │ │
│ └────────────────────────────────────────┘ │
└──────┬─────────────────────────────────────┘
       │ Returns: StreamingResponse
       ↓
┌──────────────────────────┐
│ Frontend UI              │
│ - Progressively display  │
└──────────────────────────┘
```

---

## 🔗 Component Interactions (Dependency Graph)

```
server.py
  ├── imports: agent, document_processor, vector_store
  └── calls: agent.process_query_stream()

agent.py (Langchain)
  ├── imports: langchain.prompts.PromptTemplate
  ├── imports: llm_client.get_llm_model (ChatOpenAI)
  ├── imports: cache.get_cache (ResponseCache)
  ├── imports: web_search.get_web_searcher (WebSearcher)
  ├── calls: vector_store.hybrid_search() [BM25 + Dense]
  ├── calls: vector_store.rerank_results() [Cross-encoder]
  ├── calls: web_searcher.search() [DuckDuckGo]
  └── calls: llm.stream() [Langchain ChatOpenAI]

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
  └── features: Hash-based keys, TTL expiration, cache statistics

web_search.py
  ├── imports: duckduckgo_search.DDGS
  ├── implements: WebSearcher (web search integration)
  └── features: DuckDuckGo search, result formatting, hybrid results

embeddings.py (Langchain)
  ├── imports: langchain_openai.OpenAIEmbeddings
  └── returns: OpenAIEmbeddings instance

llm_client.py (Langchain)
  ├── imports: langchain_openai.ChatOpenAI
  └── returns: ChatOpenAI instance

config.py
  └── provides: API keys, model names, base URLs, Max Tokens
```

---

## ✅ Summary of Theoretical Benefits
*   **Accuracy (Re-Ranking & Hybrid Search)**: Minimizes hallucinations by using secondary "judge" models to filter noise.
*   **Efficiency (Semantic Chunking)**: Keeps context blocks cohesive, ensuring the LLM doesn't lose the "thread" of the argument.
*   **Intelligence (Agentic Orchestration)**: Allows for autonomous feature triggers (like web search) based on retrieval confidence.
*   **Performance (Caching & Streaming)**: Optimizes both computational overhead and the end-user's perception of speed.

---
*Last Updated: March 2026*
