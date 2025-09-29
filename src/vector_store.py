"""
Vector Store Operations using ChromaDB Cloud for document storage and retrieval.

This module implements the VectorStore class for Phase 3.1 of the RAG-Enhanced Chat application,
providing cloud-based storage, similarity search, and category-based filtering.
"""

import os
import uuid
import chromadb
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store implementation using ChromaDB Cloud for document chunk storage and retrieval.

    Features:
    - Cloud-based storage (no local persistence)
    - Category-based metadata filtering
    - Similarity search with configurable top-k
    - Document management (add/delete)
    - Automatic collection management
    """

    def __init__(self, collection_name: str = "documents"):
        """
        Initialize ChromaDB Cloud client.

        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        self._client = None
        self._collection = None

        # Get cloud credentials from environment
        self.api_key = os.getenv("CHROMA_API_KEY")
        self.tenant = os.getenv("CHROMA_TENANT")
        self.database = os.getenv("CHROMA_DATABASE")

        # Validate required environment variables
        if not self.api_key:
            raise ValueError("CHROMA_API_KEY environment variable is required")
        if not self.tenant:
            raise ValueError("CHROMA_TENANT environment variable is required")
        if not self.database:
            raise ValueError("CHROMA_DATABASE environment variable is required")

        # Initialize ChromaDB Cloud client
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB Cloud client and collection"""
        try:
            # Create ChromaDB Cloud client
            self._client = chromadb.CloudClient(
                tenant=self.tenant,
                database=self.database,
                api_key=self.api_key
            )

            # Get or create collection
            try:
                self._collection = self._client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Connected to existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "RAG document chunks with category metadata"}
                )
                logger.info(f"Created new collection: {self.collection_name}")

            logger.info(f"ChromaDB Cloud initialized successfully for tenant: {self.tenant}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB Cloud: {str(e)}")
            raise Exception(f"Vector store initialization failed: {str(e)}")

    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]],
                     embeddings: List[List[float]]) -> Tuple[bool, str, List[str]]:
        """
        Store document chunks with embeddings and metadata.

        Args:
            chunks: List of text chunks
            metadata: List of metadata dicts for each chunk
            embeddings: List of embedding vectors for each chunk

        Returns:
            Tuple of (success, message, list_of_chunk_ids)
        """
        try:
            if not chunks or not metadata or not embeddings:
                return False, "Empty chunks, metadata, or embeddings provided", []

            if len(chunks) != len(metadata) or len(chunks) != len(embeddings):
                return False, "Chunks, metadata, and embeddings must have same length", []

            # Generate unique IDs for each chunk
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]

            # Prepare metadata with additional fields
            processed_metadata = []
            for i, meta in enumerate(metadata):
                processed_meta = meta.copy()
                processed_meta.update({
                    "chunk_index": i,
                    "added_at": datetime.now().isoformat(),
                    "chunk_length": len(chunks[i])
                })

                # Ensure all metadata values are strings (ChromaDB requirement)
                for key, value in processed_meta.items():
                    if not isinstance(value, (str, int, float, bool)):
                        processed_meta[key] = str(value)

                processed_metadata.append(processed_meta)

            # Filter out None embeddings
            valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]

            if not valid_indices:
                return False, "No valid embeddings found", []

            # Filter data to only include valid embeddings
            valid_chunks = [chunks[i] for i in valid_indices]
            valid_metadata = [processed_metadata[i] for i in valid_indices]
            valid_embeddings = [embeddings[i] for i in valid_indices]
            valid_ids = [chunk_ids[i] for i in valid_indices]

            # Add to collection
            self._collection.add(
                documents=valid_chunks,
                embeddings=valid_embeddings,
                metadatas=valid_metadata,
                ids=valid_ids
            )

            success_message = f"Successfully added {len(valid_chunks)} chunks to vector store"
            if len(valid_chunks) < len(chunks):
                success_message += f" (skipped {len(chunks) - len(valid_chunks)} chunks with invalid embeddings)"

            logger.info(success_message)
            return True, success_message, valid_ids

        except Exception as e:
            error_message = f"Failed to add documents to vector store: {str(e)}"
            logger.error(error_message)
            return False, error_message, []

    def similarity_search(self, query_embedding: List[float],
                         category_filter: Optional[str] = None,
                         top_k: int = 5) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Perform similarity search with optional category filtering.

        Args:
            query_embedding: Query embedding vector
            category_filter: Optional category to filter results
            top_k: Number of results to return

        Returns:
            Tuple of (success, list_of_results, error_message)
            Each result contains: {"text": str, "metadata": dict, "similarity": float, "id": str}
        """
        try:
            if not query_embedding:
                return False, [], "Empty query embedding provided"

            # Prepare where clause for category filtering
            where_clause = None
            if category_filter:
                where_clause = {"category": category_filter}

            # Perform search
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )

            # Process results
            processed_results = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
                distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
                ids = results['ids'][0] if results['ids'] else [''] * len(documents)

                for i in range(len(documents)):
                    # Convert distance to similarity (ChromaDB returns distances, lower = more similar)
                    similarity = 1.0 / (1.0 + distances[i]) if distances[i] is not None else 0.0

                    processed_results.append({
                        "text": documents[i],
                        "metadata": metadatas[i] if metadatas[i] else {},
                        "similarity": similarity,
                        "id": ids[i] if ids[i] else '',
                        "distance": distances[i] if distances[i] is not None else 0.0
                    })

            search_info = f"Found {len(processed_results)} results"
            if category_filter:
                search_info += f" for category '{category_filter}'"

            logger.info(search_info)
            return True, processed_results, None

        except Exception as e:
            error_message = f"Similarity search failed: {str(e)}"
            logger.error(error_message)
            return False, [], error_message

    def delete_document(self, document_id: str) -> Tuple[bool, str]:
        """
        Delete a specific document/chunk by ID.

        Args:
            document_id: ID of the document chunk to delete

        Returns:
            Tuple of (success, message)
        """
        try:
            if not document_id:
                return False, "Empty document ID provided"

            # Check if document exists
            existing = self._collection.get(ids=[document_id])

            if not existing['ids']:
                return False, f"Document with ID '{document_id}' not found"

            # Delete the document
            self._collection.delete(ids=[document_id])

            logger.info(f"Successfully deleted document: {document_id}")
            return True, f"Document '{document_id}' deleted successfully"

        except Exception as e:
            error_message = f"Failed to delete document '{document_id}': {str(e)}"
            logger.error(error_message)
            return False, error_message

    def delete_documents_by_category(self, category: str) -> Tuple[bool, str, int]:
        """
        Delete all documents belonging to a specific category.

        Args:
            category: Category name to filter and delete

        Returns:
            Tuple of (success, message, count_deleted)
        """
        try:
            if not category:
                return False, "Empty category provided", 0

            # Find documents in category
            results = self._collection.get(
                where={"category": category}
            )

            if not results['ids']:
                return True, f"No documents found in category '{category}'", 0

            # Delete all documents in category
            self._collection.delete(
                where={"category": category}
            )

            count_deleted = len(results['ids'])
            logger.info(f"Deleted {count_deleted} documents from category: {category}")
            return True, f"Deleted {count_deleted} documents from category '{category}'", count_deleted

        except Exception as e:
            error_message = f"Failed to delete documents from category '{category}': {str(e)}"
            logger.error(error_message)
            return False, error_message, 0

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get collection count
            count_result = self._collection.count()

            # Get sample of metadata to understand categories
            sample_results = self._collection.get(limit=100)

            categories = set()
            if sample_results['metadatas']:
                for meta in sample_results['metadatas']:
                    if 'category' in meta:
                        categories.add(meta['category'])

            return {
                "collection_name": self.collection_name,
                "total_documents": count_result,
                "categories_found": list(categories),
                "tenant": self.tenant,
                "database": self.database,
                "initialized": self._client is not None
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "error": str(e),
                "initialized": False
            }

    def clear_collection(self) -> Tuple[bool, str]:
        """
        Clear all documents from the collection.

        Returns:
            Tuple of (success, message)
        """
        try:
            # Get all document IDs
            all_docs = self._collection.get()

            if not all_docs['ids']:
                return True, "Collection is already empty"

            count_before = len(all_docs['ids'])

            # Delete all documents
            self._collection.delete(
                ids=all_docs['ids']
            )

            logger.info(f"Cleared {count_before} documents from collection")
            return True, f"Successfully cleared {count_before} documents from collection"

        except Exception as e:
            error_message = f"Failed to clear collection: {str(e)}"
            logger.error(error_message)
            return False, error_message

    def health_check(self) -> Tuple[bool, str]:
        """
        Check if the vector store is healthy and accessible.

        Returns:
            Tuple of (healthy, status_message)
        """
        try:
            if not self._client or not self._collection:
                return False, "ChromaDB client or collection not initialized"

            # Test basic operations
            count = self._collection.count()

            return True, f"Vector store is healthy. Contains {count} documents."

        except Exception as e:
            return False, f"Vector store health check failed: {str(e)}"


# Utility functions for vector store operations

def create_vector_store(collection_name: str = "documents") -> VectorStore:
    """
    Factory function to create a VectorStore instance.

    Args:
        collection_name: ChromaDB collection name

    Returns:
        Initialized VectorStore instance
    """
    return VectorStore(collection_name)


def test_vector_store(vector_store: VectorStore) -> bool:
    """
    Test vector store with sample data.

    Args:
        vector_store: VectorStore instance to test

    Returns:
        True if all tests pass, False otherwise
    """
    try:
        # Test data
        test_chunks = ["This is a test document.", "This is another test chunk."]
        test_metadata = [
            {"category": "test", "document_id": "test_doc_1", "section": "intro"},
            {"category": "test", "document_id": "test_doc_1", "section": "body"}
        ]
        test_embeddings = [
            [0.1, 0.2, 0.3, 0.4, 0.5],  # Dummy embeddings for testing
            [0.2, 0.3, 0.4, 0.5, 0.6]
        ]

        # Test add documents
        success, message, chunk_ids = vector_store.add_documents(
            test_chunks, test_metadata, test_embeddings
        )

        if not success:
            logger.error(f"Add documents test failed: {message}")
            return False

        # Test similarity search
        success, results, error = vector_store.similarity_search(
            [0.15, 0.25, 0.35, 0.45, 0.55], category_filter="test", top_k=2
        )

        if not success:
            logger.error(f"Similarity search test failed: {error}")
            return False

        # Test delete documents
        if chunk_ids:
            success, message = vector_store.delete_document(chunk_ids[0])
            if not success:
                logger.error(f"Delete document test failed: {message}")
                return False

        logger.info("Vector store tests passed successfully")
        return True

    except Exception as e:
        logger.error(f"Vector store test failed: {str(e)}")
        return False