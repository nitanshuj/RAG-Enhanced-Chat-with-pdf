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
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from src.embeddings import get_embeddings_model
from src.logger import get_logger
from src.exception import VectorStoreException
import sys

load_dotenv()
logger = get_logger(__name__)

# Optional imports for re-ranking
try:
    from sentence_transformers import CrossEncoder
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False
    logger.warning("sentence-transformers not available. Re-ranking disabled.")


class VectorStore:
    """
    Vector store implementation using Langchain Chroma Cloud.

    Features:
    - Cloud-based storage
    - Category-based metadata filtering
    - Similarity search
    - Document management
    """

    def __init__(self, collection_name: str = "documents",
                 enable_hybrid_search: bool = True,
                 enable_reranking: bool = True):
        """
        Initialize Langchain Chroma vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            enable_hybrid_search: Enable BM25 + Dense vector hybrid search (default: True)
            enable_reranking: Enable cross-encoder re-ranking (default: True)
        """
        self.collection_name = collection_name
        self.enable_hybrid_search = enable_hybrid_search
        self.enable_reranking = enable_reranking and RERANKING_AVAILABLE
        self.bm25_retriever = None
        self.all_documents = []  # Store documents for BM25 indexing
        self.cross_encoder = None

        # Initialize cross-encoder for re-ranking
        if self.enable_reranking:
            try:
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Cross-encoder loaded for re-ranking")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
                self.enable_reranking = False

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
            if enable_hybrid_search:
                logger.info("Hybrid search (BM25 + Dense) enabled")
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

            # Update BM25 index if hybrid search is enabled
            if self.enable_hybrid_search:
                self.all_documents.extend(documents)
                self._update_bm25_retriever()

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
                         section_filter: Optional[str] = None,
                         min_confidence: Optional[float] = None,
                         metadata_filters: Optional[Dict[str, Any]] = None,
                         top_k: int = 5) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Perform similarity search using Langchain with enhanced metadata filtering.

        Args:
            query_embedding: Query embedding vector (optional if query_text provided)
            query_text: Query text (Langchain will embed it)
            category_filter: Optional category to filter results
            section_filter: Optional section name to filter results
            min_confidence: Minimum confidence score for results
            metadata_filters: Additional custom metadata filters
            top_k: Number of results to return

        Returns:
            Tuple of (success, list_of_results, error_message)
        """
        try:
            if not query_text and not query_embedding:
                return False, [], "Either query_text or query_embedding required"

            # Build comprehensive filter
            filter_dict = {}

            if category_filter:
                filter_dict["category"] = category_filter

            if section_filter:
                filter_dict["section"] = section_filter

            # Add custom metadata filters
            if metadata_filters:
                filter_dict.update(metadata_filters)

            # Convert to None if empty (Chroma requirement)
            if not filter_dict:
                filter_dict = None

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

            # Convert to expected format and apply confidence filtering
            processed_results = []
            for doc in docs:
                # Apply confidence filter if specified
                if min_confidence is not None:
                    doc_confidence = doc.metadata.get("confidence", 1.0)
                    if doc_confidence < min_confidence:
                        continue

                processed_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": 0.8,  # Langchain doesn't always return scores
                    "id": doc.metadata.get("id", ""),
                })

            logger.info(f"Found {len(processed_results)} results (after filtering)")
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

    def _update_bm25_retriever(self):
        """Update BM25 retriever with current documents."""
        try:
            if not self.all_documents:
                logger.warning("No documents available for BM25 indexing")
                return

            self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
            self.bm25_retriever.k = 5  # Default top_k for BM25
            logger.info(f"BM25 retriever updated with {len(self.all_documents)} documents")
        except Exception as e:
            logger.error(f"Failed to update BM25 retriever: {str(e)}")
            self.bm25_retriever = None

    def hybrid_search(self, query_text: str,
                     category_filter: Optional[str] = None,
                     section_filter: Optional[str] = None,
                     min_confidence: Optional[float] = None,
                     metadata_filters: Optional[Dict[str, Any]] = None,
                     top_k: int = 5,
                     bm25_weight: float = 0.5) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Perform hybrid search combining BM25 (keyword) and dense vector (semantic) search.

        Args:
            query_text: Query text
            category_filter: Optional category to filter results
            section_filter: Optional section name to filter results
            min_confidence: Minimum confidence score for results
            metadata_filters: Additional custom metadata filters
            top_k: Number of results to return
            bm25_weight: Weight for BM25 results (0.0-1.0), dense weight = 1 - bm25_weight

        Returns:
            Tuple of (success, list_of_results, error_message)
        """
        try:
            if not query_text:
                return False, [], "Query text is required"

            # If hybrid search is disabled or BM25 not available, fall back to dense search
            if not self.enable_hybrid_search or not self.bm25_retriever:
                logger.info("Hybrid search not available, using dense search only")
                return self.similarity_search(
                    query_text=query_text,
                    category_filter=category_filter,
                    section_filter=section_filter,
                    min_confidence=min_confidence,
                    metadata_filters=metadata_filters,
                    top_k=top_k
                )

            # Get dense retriever from vectorstore
            dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})

            # Build comprehensive filter
            filter_dict = {}
            if category_filter:
                filter_dict["category"] = category_filter
            if section_filter:
                filter_dict["section"] = section_filter
            if metadata_filters:
                filter_dict.update(metadata_filters)

            # Apply filters to dense retriever
            if filter_dict:
                dense_retriever.search_kwargs["filter"] = filter_dict

            # Create ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, dense_retriever],
                weights=[bm25_weight, 1.0 - bm25_weight]
            )

            # Perform hybrid search
            docs = ensemble_retriever.invoke(query_text)

            # Apply all filters to BM25 results
            filtered_docs = []
            for doc in docs:
                # Category filter
                if category_filter and doc.metadata.get("category") != category_filter:
                    continue
                # Section filter
                if section_filter and doc.metadata.get("section") != section_filter:
                    continue
                # Confidence filter
                if min_confidence is not None:
                    doc_confidence = doc.metadata.get("confidence", 1.0)
                    if doc_confidence < min_confidence:
                        continue
                # Custom metadata filters
                if metadata_filters:
                    skip = False
                    for key, value in metadata_filters.items():
                        if doc.metadata.get(key) != value:
                            skip = True
                            break
                    if skip:
                        continue

                filtered_docs.append(doc)

            # Limit to top_k results
            filtered_docs = filtered_docs[:top_k]

            # Convert to expected format
            processed_results = []
            for doc in filtered_docs:
                processed_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": 0.8,  # Ensemble doesn't return scores
                    "id": doc.metadata.get("id", ""),
                })

            logger.info(f"Hybrid search found {len(processed_results)} results (after filtering)")
            return True, processed_results, None

        except Exception as e:
            error_message = f"Hybrid search failed: {str(e)}"
            logger.error(error_message, exc_info=True)
            return False, [], error_message

    def rerank_results(self, query_text: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Re-rank search results using cross-encoder model.

        Args:
            query_text: Original query
            results: List of retrieved documents
            top_k: Number of top results to return after re-ranking

        Returns:
            Re-ranked list of documents
        """
        if not self.enable_reranking or not self.cross_encoder or not results:
            return results[:top_k]

        try:
            # Prepare query-document pairs for cross-encoder
            pairs = [[query_text, doc["text"]] for doc in results]

            # Get scores from cross-encoder
            scores = self.cross_encoder.predict(pairs)

            # Add scores to results and sort
            for doc, score in zip(results, scores):
                doc["rerank_score"] = float(score)
                doc["similarity"] = float(score)  # Update similarity with rerank score

            # Sort by rerank score (descending)
            reranked_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

            logger.info(f"Re-ranked {len(results)} results, returning top {top_k}")
            return reranked_results[:top_k]

        except Exception as e:
            logger.error(f"Re-ranking failed: {str(e)}")
            return results[:top_k]


def create_vector_store(collection_name: str = "documents") -> VectorStore:
    """Factory function to create a VectorStore instance."""
    return VectorStore(collection_name)
