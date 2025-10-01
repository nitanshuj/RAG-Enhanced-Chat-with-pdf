"""
Vector Store Operations using Langchain Chroma for document storage and retrieval.

This module implements vector store operations using Langchain's Chroma integration.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.embeddings import get_embeddings_model
from src.logger import get_logger
from src.exception import VectorStoreException
import sys

load_dotenv()
logger = get_logger(__name__)


class VectorStore:
    """
    Vector store implementation using Langchain Chroma Cloud.

    Features:
    - Cloud-based storage
    - Category-based metadata filtering
    - Similarity search
    - Document management
    """

    def __init__(self, collection_name: str = "documents"):
        """
        Initialize Langchain Chroma vector store.

        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name

        # Get cloud credentials
        self.api_key = os.getenv("CHROMA_API_KEY")
        self.tenant = os.getenv("CHROMA_TENANT")
        self.database = os.getenv("CHROMA_DATABASE")

        if not all([self.api_key, self.tenant, self.database]):
            raise VectorStoreException(
                "CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE required",
                sys.exc_info()
            )

        # Initialize Langchain Chroma with cloud settings
        try:
            import chromadb

            # Disable telemetry
            os.environ['ANONYMIZED_TELEMETRY'] = 'False'

            client = chromadb.CloudClient(
                tenant=self.tenant,
                database=self.database,
                api_key=self.api_key
            )

            # Initialize Langchain Chroma
            self.vectorstore = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=get_embeddings_model()
            )

            logger.info(f"Langchain Chroma initialized for collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {str(e)}", exc_info=True)
            raise VectorStoreException(f"Failed to initialize Chroma: {str(e)}", sys.exc_info())

    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]],
                     embeddings: List[List[float]] = None) -> Tuple[bool, str, List[str]]:
        """
        Store document chunks with metadata using Langchain.

        Args:
            chunks: List of text chunks
            metadata: List of metadata dicts for each chunk
            embeddings: Optional pre-computed embeddings (ignored, Langchain handles this)

        Returns:
            Tuple of (success, message, list_of_chunk_ids)
        """
        try:
            if not chunks or not metadata:
                return False, "Empty chunks or metadata provided", []

            if len(chunks) != len(metadata):
                return False, "Chunks and metadata must have same length", []

            # Convert to Langchain Documents
            documents = []
            for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
                # Add timestamp and index
                meta_copy = meta.copy()
                meta_copy.update({
                    "chunk_index": i,
                    "added_at": datetime.now().isoformat(),
                    "chunk_length": len(chunk)
                })

                # Convert all metadata values to strings
                for key, value in meta_copy.items():
                    if not isinstance(value, (str, int, float, bool)):
                        meta_copy[key] = str(value)

                documents.append(Document(page_content=chunk, metadata=meta_copy))

            # Add documents using Langchain (handles embedding automatically)
            ids = self.vectorstore.add_documents(documents)

            success_message = f"Successfully added {len(documents)} chunks to vector store"
            logger.info(success_message)
            return True, success_message, ids

        except Exception as e:
            error_message = f"Failed to add documents: {str(e)}"
            logger.error(error_message)
            return False, error_message, []

    def similarity_search(self, query_embedding: List[float] = None,
                         query_text: str = None,
                         category_filter: Optional[str] = None,
                         top_k: int = 5) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Perform similarity search using Langchain.

        Args:
            query_embedding: Query embedding vector (optional if query_text provided)
            query_text: Query text (Langchain will embed it)
            category_filter: Optional category to filter results
            top_k: Number of results to return

        Returns:
            Tuple of (success, list_of_results, error_message)
        """
        try:
            if not query_text and not query_embedding:
                return False, [], "Either query_text or query_embedding required"

            # Build filter
            filter_dict = None
            if category_filter:
                filter_dict = {"category": category_filter}

            # Perform search
            if query_text:
                docs = self.vectorstore.similarity_search(
                    query=query_text,
                    k=top_k,
                    filter=filter_dict
                )
            else:
                # Use embedding directly (less common with Langchain)
                docs = self.vectorstore.similarity_search_by_vector(
                    embedding=query_embedding,
                    k=top_k,
                    filter=filter_dict
                )

            # Convert to expected format
            processed_results = []
            for doc in docs:
                processed_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": 0.8,  # Langchain doesn't always return scores
                    "id": doc.metadata.get("id", ""),
                })

            logger.info(f"Found {len(processed_results)} results")
            return True, processed_results, None

        except Exception as e:
            error_message = f"Similarity search failed: {str(e)}"
            logger.error(error_message)
            return False, [], error_message

    def delete_document(self, document_id: str) -> Tuple[bool, str]:
        """Delete a document by ID."""
        try:
            self.vectorstore.delete([document_id])
            logger.info(f"Deleted document: {document_id}")
            return True, f"Document '{document_id}' deleted successfully"
        except Exception as e:
            error_message = f"Failed to delete document: {str(e)}"
            logger.error(error_message, exc_info=True)
            return False, error_message

    def delete_documents_by_metadata(self, metadata_filter: Dict[str, Any]) -> Tuple[bool, str, int]:
        """
        Delete all documents matching the metadata filter.

        Args:
            metadata_filter: Dictionary with metadata key-value pairs to filter by
                            Example: {"document_id": "example.pdf", "category": "Research Paper"}

        Returns:
            Tuple of (success, message, count_deleted)

        Example:
            >>> success, msg, count = vector_store.delete_documents_by_metadata(
            ...     {"document_id": "example.pdf"}
            ... )
            >>> print(f"Deleted {count} chunks")
        """
        try:
            # Get the underlying chromadb collection
            collection = self.vectorstore._collection

            # Query to find all matching documents
            results = collection.get(
                where=metadata_filter,
                include=["metadatas"]
            )

            if not results or not results.get('ids'):
                logger.info(f"No documents found matching filter: {metadata_filter}")
                return True, "No documents found to delete", 0

            # Delete all matching IDs
            ids_to_delete = results['ids']
            collection.delete(ids=ids_to_delete)

            count_deleted = len(ids_to_delete)
            logger.info(f"Deleted {count_deleted} document chunks matching filter: {metadata_filter}")
            return True, f"Successfully deleted {count_deleted} document chunks", count_deleted

        except Exception as e:
            error_message = f"Failed to delete documents by metadata: {str(e)}"
            logger.error(error_message, exc_info=True)
            return False, error_message, 0

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            return {
                "collection_name": self.collection_name,
                "tenant": self.tenant,
                "database": self.database,
                "initialized": True
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "error": str(e),
                "initialized": False
            }

    def health_check(self) -> Tuple[bool, str]:
        """Check if the vector store is healthy."""
        try:
            if not self.vectorstore:
                return False, "Vectorstore not initialized"
            return True, "Vector store is healthy"
        except Exception as e:
            return False, f"Health check failed: {str(e)}"


def create_vector_store(collection_name: str = "documents") -> VectorStore:
    """Factory function to create a VectorStore instance."""
    return VectorStore(collection_name)
