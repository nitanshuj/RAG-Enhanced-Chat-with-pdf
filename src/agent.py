"""
RAG Agent using Langchain for retrieval and generation pipeline.

This module implements a RAG agent using Langchain chains for streamlined processing.
"""

from typing import List, Dict, Any, Optional, Tuple, Iterator
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.memory import VectorStoreRetrieverMemory
from langchain_chroma import Chroma
from src.llm_client import get_llm_model
from src.logger import get_logger
from src.exception import AgentException
from src.cache import get_cache
from src.embeddings import get_embeddings_model
from src.web_search import get_web_searcher
import sys
import os

logger = get_logger(__name__)


class SimpleRAGAgent:
    """
    RAG Agent using Langchain for document retrieval and answer generation.

    Features:
    - Langchain-powered retrieval chains
    - Automatic query processing
    - Context formatting
    - Session history management
    """

    def __init__(self, llm_client, vector_store,
                 use_hybrid_search: bool = True,
                 use_reranking: bool = True,
                 use_caching: bool = True,
                 cache_ttl: int = 3600,
                 use_semantic_memory: bool = True,
                 use_web_search: bool = True):
        """
        Initialize the RAG agent with Langchain components.

        Args:
            llm_client: LLM client module (for backwards compatibility)
            vector_store: VectorStore instance with Langchain Chroma
            use_hybrid_search: Enable hybrid search (BM25 + Dense vectors)
            use_reranking: Enable cross-encoder re-ranking
            use_caching: Enable response caching
            cache_ttl: Cache time-to-live in seconds (default: 3600 = 1 hour)
            use_semantic_memory: Enable semantic memory for conversation history
            use_web_search: Enable hybrid knowledge retrieval (85% docs + 15% web)
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.llm = get_llm_model()
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        self.use_caching = use_caching
        self.cache = get_cache(ttl_seconds=cache_ttl) if use_caching else None
        self.use_semantic_memory = use_semantic_memory
        self.semantic_memory = None
        self.use_web_search = use_web_search
        self.web_searcher = get_web_searcher(enabled=use_web_search) if use_web_search else None

        # Initialize semantic memory
        if use_semantic_memory:
            try:
                self._initialize_semantic_memory()
            except Exception as e:
                logger.warning(f"Failed to initialize semantic memory: {e}")
                self.use_semantic_memory = False

        # Session state
        self.conversation_history = []
        self.session_metadata = {
            "session_start": datetime.now().isoformat(),
            "total_queries": 0,
            "categories_queried": set()
        }

        # Create custom prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert document analysis assistant. 
                            Use the following context to answer the question.
                            If the information is not available in the context, clearly state that.
                            
                            Context: {context}
                            Question: {question}
                            Answer:
        """
        )

        logger.info(f"SimpleRAGAgent initialized (Hybrid: {use_hybrid_search}, Reranking: {use_reranking}, Caching: {use_caching}, Memory: {use_semantic_memory}, WebSearch: {use_web_search})")

    def _initialize_semantic_memory(self):
        """Initialize semantic memory for conversation history."""
        import chromadb

        # Create a separate collection for conversation memory
        client = chromadb.CloudClient(
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
            api_key=os.getenv("CHROMA_API_KEY")
        )

        # Create memory vector store
        memory_vectorstore = Chroma(
            client=client,
            collection_name="conversation_memory",
            embedding_function=get_embeddings_model()
        )

        # Create semantic memory retriever
        retriever = memory_vectorstore.as_retriever(search_kwargs={"k": 3})

        self.semantic_memory = VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key="history",
            input_key="input",
            output_key="output"
        )

        logger.info("Semantic memory initialized for conversation history")

    def _get_relevant_memory(self, query: str) -> str:
        """
        Retrieve relevant past interactions from semantic memory.

        Args:
            query: Current query

        Returns:
            Formatted string of relevant past interactions
        """
        if not self.use_semantic_memory or not self.semantic_memory:
            return ""

        try:
            # Load relevant memory context
            memory_vars = self.semantic_memory.load_memory_variables({"input": query})
            history = memory_vars.get("history", "")

            if history:
                return f"\n\nRelevant past conversation:\n{history}\n"
            return ""

        except Exception as e:
            logger.warning(f"Failed to retrieve semantic memory: {e}")
            return ""

    def _save_to_memory(self, query: str, answer: str):
        """
        Save interaction to semantic memory.

        Args:
            query: User query
            answer: Assistant answer
        """
        if not self.use_semantic_memory or not self.semantic_memory:
            return

        try:
            self.semantic_memory.save_context(
                {"input": query},
                {"output": answer}
            )
            logger.debug("Interaction saved to semantic memory")
        except Exception as e:
            logger.warning(f"Failed to save to semantic memory: {e}")

    def process_query(self, query: str,
                      category_filter: Optional[str] = None,
                      top_k: int = 5,
                      include_history: bool = True,
                      document_id: Optional[str] = None) -> Tuple[bool, Dict[str, Any], str]:
        """
        Process user query through Langchain RAG pipeline.

        Args:
            query: User question/query
            category_filter: Optional category to filter documents
            top_k: Number of top documents to retrieve
            include_history: Whether to include conversation history
            document_id: Optional document identifier for caching

        Returns:
            Tuple of (success, response_data, error_message)
        """
        try:
            # Validate query
            if not query or not query.strip():
                return False, {}, "Empty query provided"

            processed_query = query.strip()

            # Check cache first
            if self.use_caching and self.cache:
                cached_response = self.cache.get(
                    query=processed_query,
                    document_id=document_id,
                    category_filter=category_filter
                )
                if cached_response:
                    logger.info("Returning cached response")
                    return True, cached_response, None

            # Retrieve relevant documents using hybrid or dense search
            # Retrieve more documents initially if re-ranking is enabled
            retrieval_k = top_k * 2 if self.use_reranking else top_k

            if self.use_hybrid_search and hasattr(self.vector_store, 'hybrid_search'):
                success, retrieved_docs, error = self.vector_store.hybrid_search(
                    query_text=processed_query,
                    category_filter=category_filter,
                    top_k=retrieval_k,
                    bm25_weight=0.5  # 50% BM25, 50% dense vectors
                )
            else:
                success, retrieved_docs, error = self.vector_store.similarity_search(
                    query_text=processed_query,
                    category_filter=category_filter,
                    top_k=retrieval_k
                )

            if not success:
                return False, {}, f"Document retrieval failed: {error}"

            # Apply re-ranking if enabled
            if self.use_reranking and hasattr(self.vector_store, 'rerank_results'):
                retrieved_docs = self.vector_store.rerank_results(
                    query_text=processed_query,
                    results=retrieved_docs,
                    top_k=top_k
                )

            # Check if web search should be triggered
            web_results = []
            if self.use_web_search and self.web_searcher:
                should_search_web = self.web_searcher.should_use_web_search(
                    doc_results=retrieved_docs,
                    query=processed_query,
                    explicit_request=False  # Can be parameterized later
                )

                if should_search_web:
                    web_results = self.web_searcher.search(processed_query, max_results=2)

                    # Combine document and web results (85% docs, 15% web)
                    if web_results:
                        retrieved_docs = self.web_searcher.combine_results(
                            doc_results=retrieved_docs,
                            web_results=web_results,
                            doc_weight=0.85
                        )

            # Format context
            formatted_context = self._format_context(retrieved_docs)

            # Add web context if available
            if web_results:
                web_context = self.web_searcher.format_web_context(web_results)
                formatted_context += web_context

            # Get relevant memory if enabled
            memory_context = self._get_relevant_memory(processed_query)

            # Generate answer using LLM
            prompt = self.prompt_template.format(
                context=formatted_context + memory_context,
                question=processed_query
            )


            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content

            # Save to semantic memory
            self._save_to_memory(processed_query, answer)

            # Update session state
            self._update_session_state(processed_query, answer, category_filter, retrieved_docs)

            # Prepare response
            response_data = {
                "answer": answer,
                "sources": retrieved_docs,
                "context_used": formatted_context,
                "query_processed": processed_query,
                "category_filter": category_filter,
                "documents_found": len(retrieved_docs),
                "web_search_used": len(web_results) > 0,
                "web_results_count": len(web_results)
            }

            # Cache the response
            if self.use_caching and self.cache:
                self.cache.set(
                    query=processed_query,
                    response=response_data,
                    document_id=document_id,
                    category_filter=category_filter
                )

            logger.info(f"Query processed successfully. Found {len(retrieved_docs)} documents")
            return True, response_data, None

        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            logger.error(error_message, exc_info=True)
            return False, {}, error_message

    def _format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into context string."""
        if not retrieved_docs:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            category = metadata.get("category", "Unknown")
            section = metadata.get("section", "")

            doc_header = f"Document {i}"
            if category != "Unknown":
                doc_header += f" (Category: {category}"
                if section:
                    doc_header += f", Section: {section}"
                doc_header += ")"

            context_parts.append(f"{doc_header}:\n{text}")

        formatted_context = "\n\n---\n\n".join(context_parts)

        # Truncate if too long
        max_length = 2500
        if len(formatted_context) > max_length:
            formatted_context = formatted_context[:max_length] + "\n\n[Context truncated...]"

        return formatted_context

    def _update_session_state(self, query: str, answer: str, category_filter: Optional[str],
                             retrieved_docs: List[Dict]):
        """Update session state with current interaction."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "category_filter": category_filter,
            "documents_count": len(retrieved_docs),
        }

        self.conversation_history.append(interaction)

        # Keep only recent history
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        self.session_metadata["total_queries"] += 1
        if category_filter:
            self.session_metadata["categories_queried"].add(category_filter)

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        info = {
            "session_start": self.session_metadata["session_start"],
            "total_queries": self.session_metadata["total_queries"],
            "categories_queried": list(self.session_metadata["categories_queried"]),
            "conversation_length": len(self.conversation_history),
            "vector_store_info": self.vector_store.get_collection_info()
        }

        # Add cache statistics if caching is enabled
        if self.use_caching and self.cache:
            info["cache_stats"] = self.cache.get_stats()

        return info

    def clear_session(self):
        """Clear conversation history and reset session state."""
        self.conversation_history = []
        self.session_metadata = {
            "session_start": datetime.now().isoformat(),
            "total_queries": 0,
            "categories_queried": set()
        }
        logger.info("Session cleared")

    def export_conversation(self) -> List[Dict[str, Any]]:
        """Export conversation history."""
        return self.conversation_history.copy()

    def process_query_stream(self, query: str,
                            category_filter: Optional[str] = None,
                            top_k: int = 5) -> Iterator[str]:
        """
        Process query and stream the response token by token.

        Args:
            query: User question/query
            category_filter: Optional category to filter documents
            top_k: Number of top documents to retrieve

        Yields:
            Response tokens as they are generated
        """
        try:
            # Validate query
            if not query or not query.strip():
                yield "Error: Empty query provided"
                return

            processed_query = query.strip()

            # Retrieve relevant documents (same as non-streaming)
            retrieval_k = top_k * 2 if self.use_reranking else top_k

            if self.use_hybrid_search and hasattr(self.vector_store, 'hybrid_search'):
                success, retrieved_docs, error = self.vector_store.hybrid_search(
                    query_text=processed_query,
                    category_filter=category_filter,
                    top_k=retrieval_k,
                    bm25_weight=0.5
                )
            else:
                success, retrieved_docs, error = self.vector_store.similarity_search(
                    query_text=processed_query,
                    category_filter=category_filter,
                    top_k=retrieval_k
                )

            if not success:
                yield f"Error: Document retrieval failed - {error}"
                return

            # Apply re-ranking if enabled
            if self.use_reranking and hasattr(self.vector_store, 'rerank_results'):
                retrieved_docs = self.vector_store.rerank_results(
                    query_text=processed_query,
                    results=retrieved_docs,
                    top_k=top_k
                )

            # Format context
            formatted_context = self._format_context(retrieved_docs)

            # Generate streaming answer
            prompt = self.prompt_template.format(
                context=formatted_context,
                question=processed_query
            )

            # Stream tokens from LLM
            full_response = ""
            for chunk in self.llm.stream([HumanMessage(content=prompt)]):
                if hasattr(chunk, 'content'):
                    token = chunk.content
                    full_response += token
                    yield token

            # Update session state after streaming completes
            self._update_session_state(processed_query, full_response, category_filter, retrieved_docs)

        except Exception as e:
            error_message = f"Error processing streaming query: {str(e)}"
            logger.error(error_message, exc_info=True)
            yield f"Error: {error_message}"


def create_rag_agent(vector_store) -> SimpleRAGAgent:
    """Factory function to create a RAG agent."""
    import src.llm_client as llm_client
    return SimpleRAGAgent(llm_client, vector_store)
