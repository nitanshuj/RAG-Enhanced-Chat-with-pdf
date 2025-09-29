"""
Simple RAG Agent for Phase 4.1 - Basic RAG Implementation

This module implements the SimpleRAGAgent class that orchestrates the complete
RAG pipeline: query processing, document retrieval, context formatting, and response generation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRAGAgent:
    """
    Simple RAG Agent that implements the basic retrieve-then-generate pipeline.

    Features:
    - Query processing and validation
    - Context retrieval with category filtering
    - Context formatting for LLM consumption
    - Response generation with basic prompt engineering
    - Simple session state management
    """

    def __init__(self, llm_client, vector_store):
        """
        Initialize the RAG agent with required components.

        Args:
            llm_client: LLM client for embeddings and response generation
            vector_store: Vector store for document retrieval
        """
        self.llm_client = llm_client
        self.vector_store = vector_store

        # Session state for conversation history
        self.conversation_history = []
        self.session_metadata = {
            "session_start": datetime.now().isoformat(),
            "total_queries": 0,
            "categories_queried": set()
        }

        logger.info("SimpleRAGAgent initialized successfully")

    def process_query(self, query: str, category_filter: Optional[str] = None,
                     top_k: int = 5, include_history: bool = True) -> Tuple[bool, Dict[str, Any], str]:
        """
        Process user query through the complete RAG pipeline.

        Args:
            query: User question/query
            category_filter: Optional category to filter documents
            top_k: Number of top documents to retrieve
            include_history: Whether to include conversation history in response

        Returns:
            Tuple of (success, response_data, error_message)
            response_data contains: {"answer": str, "sources": List[Dict], "context_used": str}
        """
        try:
            # Step 1: Validate and preprocess query
            success, processed_query, error = self._preprocess_query(query)
            if not success:
                return False, {}, error

            # Step 2: Generate query embedding
            success, query_embedding, error = self._generate_query_embedding(processed_query)
            if not success:
                return False, {}, error

            # Step 3: Retrieve relevant documents
            success, retrieved_docs, error = self._retrieve_documents(
                query_embedding, category_filter, top_k
            )
            if not success:
                return False, {}, error

            # Step 4: Format context for LLM
            formatted_context = self._format_context(retrieved_docs)

            # Step 5: Generate response
            success, answer, error = self._generate_answer(
                processed_query, formatted_context, include_history
            )
            if not success:
                return False, {}, error

            # Step 6: Update session state
            self._update_session_state(processed_query, answer, category_filter, retrieved_docs)

            # Prepare response data
            response_data = {
                "answer": answer,
                "sources": retrieved_docs,
                "context_used": formatted_context,
                "query_processed": processed_query,
                "category_filter": category_filter,
                "documents_found": len(retrieved_docs)
            }

            logger.info(f"Query processed successfully. Found {len(retrieved_docs)} relevant documents")
            return True, response_data, None

        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            logger.error(error_message)
            return False, {}, error_message

    def _preprocess_query(self, query: str) -> Tuple[bool, str, str]:
        """
        Validate and preprocess the user query.

        Args:
            query: Raw user query

        Returns:
            Tuple of (success, processed_query, error_message)
        """
        if not query or not query.strip():
            return False, "", "Empty query provided"

        # Basic preprocessing
        processed_query = query.strip()

        # Remove excessive whitespace
        processed_query = " ".join(processed_query.split())

        # Basic validation
        if len(processed_query) < 3:
            return False, "", "Query too short (minimum 3 characters)"

        if len(processed_query) > 1000:
            processed_query = processed_query[:1000]
            logger.warning("Query truncated to 1000 characters")

        return True, processed_query, None

    def _generate_query_embedding(self, query: str) -> Tuple[bool, List[float], str]:
        """
        Generate embedding for the query using the LLM client.

        Args:
            query: Processed query text

        Returns:
            Tuple of (success, embedding_vector, error_message)
        """
        try:
            # Import the embedding function from embeddings module
            from src.embeddings import generate_embeddings

            success, embedding, error = generate_embeddings(query)

            if not success:
                return False, None, f"Failed to generate query embedding: {error}"

            return True, embedding, None

        except Exception as e:
            return False, None, f"Error generating query embedding: {str(e)}"

    def _retrieve_documents(self, query_embedding: List[float],
                           category_filter: Optional[str], top_k: int) -> Tuple[bool, List[Dict], str]:
        """
        Retrieve relevant documents from vector store.

        Args:
            query_embedding: Query embedding vector
            category_filter: Optional category filter
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (success, retrieved_documents, error_message)
        """
        try:
            success, results, error = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                category_filter=category_filter,
                top_k=top_k
            )

            if not success:
                return False, [], f"Document retrieval failed: {error}"

            if not results:
                return True, [], None  # No documents found, but not an error

            return True, results, None

        except Exception as e:
            return False, [], f"Error during document retrieval: {str(e)}"

    def _format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context for LLM.

        Args:
            retrieved_docs: List of retrieved document chunks with metadata

        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documents found."

        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            similarity = doc.get("similarity", 0.0)

            # Extract useful metadata
            category = metadata.get("category", "Unknown")
            section = metadata.get("section", "")
            document_id = metadata.get("document_id", "")

            # Format document context
            doc_header = f"Document {i}"
            if category != "Unknown":
                doc_header += f" (Category: {category}"
                if section:
                    doc_header += f", Section: {section}"
                doc_header += ")"

            doc_context = f"{doc_header}:\n{text}"
            context_parts.append(doc_context)

        # Join all contexts
        formatted_context = "\n\n---\n\n".join(context_parts)

        # Truncate if too long (leave room for query and response)
        max_context_length = 2500
        if len(formatted_context) > max_context_length:
            formatted_context = formatted_context[:max_context_length] + "\n\n[Context truncated due to length...]"

        return formatted_context

    def _generate_answer(self, query: str, context: str, include_history: bool) -> Tuple[bool, str, str]:
        """
        Generate final answer using LLM with query and context.

        Args:
            query: User query
            context: Formatted context from retrieved documents
            include_history: Whether to include conversation history

        Returns:
            Tuple of (success, answer, error_message)
        """
        try:
            # Import response generation from llm_client
            from src.llm_client import generate_response

            # Enhance query with conversation history if requested
            enhanced_query = query
            if include_history and self.conversation_history:
                recent_history = self._get_recent_history(max_entries=3)
                if recent_history:
                    enhanced_query = f"Conversation history:\n{recent_history}\n\nCurrent question: {query}"

            # Generate response
            success, answer, error = generate_response(
                prompt=enhanced_query,
                context=context,
                max_tokens=800,
                temperature=0.1
            )

            if not success:
                return False, "", f"Failed to generate response: {error}"

            return True, answer, None

        except Exception as e:
            return False, "", f"Error generating answer: {str(e)}"

    def _update_session_state(self, query: str, answer: str, category_filter: Optional[str],
                             retrieved_docs: List[Dict]):
        """
        Update session state with current interaction.

        Args:
            query: User query
            answer: Generated answer
            category_filter: Category filter used
            retrieved_docs: Documents retrieved for this query
        """
        # Add to conversation history
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "category_filter": category_filter,
            "documents_count": len(retrieved_docs),
            "sources": [doc.get("metadata", {}).get("document_id", "") for doc in retrieved_docs]
        }

        self.conversation_history.append(interaction)

        # Keep only recent history (last 10 interactions)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        # Update session metadata
        self.session_metadata["total_queries"] += 1
        if category_filter:
            self.session_metadata["categories_queried"].add(category_filter)

    def _get_recent_history(self, max_entries: int = 3) -> str:
        """
        Get recent conversation history as formatted string.

        Args:
            max_entries: Maximum number of recent entries to include

        Returns:
            Formatted history string
        """
        if not self.conversation_history:
            return ""

        recent_entries = self.conversation_history[-max_entries:]
        history_parts = []

        for entry in recent_entries:
            history_part = f"Q: {entry['query']}\nA: {entry['answer']}"
            history_parts.append(history_part)

        return "\n\n".join(history_parts)

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get current session information.

        Returns:
            Dictionary with session statistics
        """
        return {
            "session_start": self.session_metadata["session_start"],
            "total_queries": self.session_metadata["total_queries"],
            "categories_queried": list(self.session_metadata["categories_queried"]),
            "conversation_length": len(self.conversation_history),
            "vector_store_info": self.vector_store.get_collection_info()
        }

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
        """
        Export conversation history.

        Returns:
            List of conversation entries
        """
        return self.conversation_history.copy()


# Utility functions for RAG agent

def create_rag_agent(vector_store) -> SimpleRAGAgent:
    """
    Factory function to create a RAG agent with default LLM client.

    Args:
        vector_store: Initialized vector store instance

    Returns:
        Configured SimpleRAGAgent instance
    """
    # Import llm_client module for the agent
    import src.llm_client as llm_client

    return SimpleRAGAgent(llm_client, vector_store)


def test_rag_agent(agent: SimpleRAGAgent, test_query: str = "What is this document about?") -> bool:
    """
    Test RAG agent with a simple query.

    Args:
        agent: SimpleRAGAgent instance
        test_query: Query to test with

    Returns:
        True if test passes, False otherwise
    """
    try:
        success, response_data, error = agent.process_query(test_query)

        if not success:
            logger.error(f"RAG agent test failed: {error}")
            return False

        logger.info("RAG agent test passed successfully")
        logger.info(f"Test response: {response_data.get('answer', '')[:100]}...")

        return True

    except Exception as e:
        logger.error(f"RAG agent test error: {str(e)}")
        return False