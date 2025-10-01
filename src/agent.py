"""
RAG Agent using Langchain for retrieval and generation pipeline.

This module implements a RAG agent using Langchain chains for streamlined processing.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from src.llm_client import get_llm_model
from src.logger import get_logger
from src.exception import AgentException
import sys

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

    def __init__(self, llm_client, vector_store):
        """
        Initialize the RAG agent with Langchain components.

        Args:
            llm_client: LLM client module (for backwards compatibility)
            vector_store: VectorStore instance with Langchain Chroma
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.llm = get_llm_model()

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

        logger.info("SimpleRAGAgent initialized with Langchain")

    def process_query(self, query: str, category_filter: Optional[str] = None,
                     top_k: int = 5, include_history: bool = True) -> Tuple[bool, Dict[str, Any], str]:
        """
        Process user query through Langchain RAG pipeline.

        Args:
            query: User question/query
            category_filter: Optional category to filter documents
            top_k: Number of top documents to retrieve
            include_history: Whether to include conversation history

        Returns:
            Tuple of (success, response_data, error_message)
        """
        try:
            # Validate query
            if not query or not query.strip():
                return False, {}, "Empty query provided"

            processed_query = query.strip()

            # Retrieve relevant documents using vector store
            success, retrieved_docs, error = self.vector_store.similarity_search(
                query_text=processed_query,
                category_filter=category_filter,
                top_k=top_k
            )

            if not success:
                return False, {}, f"Document retrieval failed: {error}"

            # Format context
            formatted_context = self._format_context(retrieved_docs)

            # Generate answer using LLM
            prompt = self.prompt_template.format(
                context=formatted_context,
                question=processed_query
            )


            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content

            # Update session state
            self._update_session_state(processed_query, answer, category_filter, retrieved_docs)

            # Prepare response
            response_data = {
                "answer": answer,
                "sources": retrieved_docs,
                "context_used": formatted_context,
                "query_processed": processed_query,
                "category_filter": category_filter,
                "documents_found": len(retrieved_docs)
            }

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
        """Export conversation history."""
        return self.conversation_history.copy()


def create_rag_agent(vector_store) -> SimpleRAGAgent:
    """Factory function to create a RAG agent."""
    import src.llm_client as llm_client
    return SimpleRAGAgent(llm_client, vector_store)
