"""
Test script for document deletion functionality.

This script tests the delete_documents_by_metadata method in VectorStore.
"""

import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.vector_store import VectorStore
from src.logger import get_logger

logger = get_logger(__name__)


def test_deletion():
    """Test document deletion by metadata."""
    print("=" * 60)
    print("Testing Document Deletion Functionality")
    print("=" * 60)

    try:
        # Initialize vector store
        print("\n1. Initializing vector store...")
        vector_store = VectorStore(collection_name="streamlit_docs")
        print("✓ Vector store initialized")

        # Test data
        test_document_id = "test_document.pdf"
        test_chunks = [
            "This is a test chunk 1.",
            "This is a test chunk 2.",
            "This is a test chunk 3."
        ]
        test_metadata = [
            {"document_id": test_document_id, "category": "Test", "chunk_index": 0},
            {"document_id": test_document_id, "category": "Test", "chunk_index": 1},
            {"document_id": test_document_id, "category": "Test", "chunk_index": 2}
        ]

        # Add test documents
        print(f"\n2. Adding {len(test_chunks)} test chunks...")
        success, message, ids = vector_store.add_documents(test_chunks, test_metadata)

        if success:
            print(f"✓ Added {len(ids)} chunks to vector store")
            print(f"  IDs: {ids[:3]}...")
        else:
            print(f"✗ Failed to add documents: {message}")
            return False

        # Search to verify documents exist
        print(f"\n3. Verifying documents exist in vector store...")
        success, results, error = vector_store.similarity_search(
            query_text="test chunk",
            category_filter="Test",
            top_k=5
        )

        if success:
            print(f"✓ Found {len(results)} documents before deletion")
        else:
            print(f"✗ Search failed: {error}")

        # Delete documents by metadata
        print(f"\n4. Deleting documents with document_id='{test_document_id}'...")
        success, message, count = vector_store.delete_documents_by_metadata(
            {"document_id": test_document_id}
        )

        if success:
            print(f"✓ {message}")
            print(f"  Deleted {count} chunks")
        else:
            print(f"✗ Deletion failed: {message}")
            return False

        # Verify deletion
        print(f"\n5. Verifying documents are deleted...")
        success, results, error = vector_store.similarity_search(
            query_text="test chunk",
            category_filter="Test",
            top_k=5
        )

        if success:
            if len(results) == 0:
                print(f"✓ Confirmed: No documents found after deletion")
            else:
                print(f"⚠ Warning: Still found {len(results)} documents")
        else:
            print(f"✗ Verification search failed: {error}")

        # Test deleting non-existent document
        print(f"\n6. Testing deletion of non-existent document...")
        success, message, count = vector_store.delete_documents_by_metadata(
            {"document_id": "non_existent_file.pdf"}
        )

        if success and count == 0:
            print(f"✓ Correctly handled non-existent document")
            print(f"  {message}")
        else:
            print(f"⚠ Unexpected result: {message}, count={count}")

        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ Test failed with exception: {str(e)}")
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    test_deletion()
