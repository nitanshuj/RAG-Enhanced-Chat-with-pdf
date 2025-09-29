"""
Simple unit tests for the VectorStore class.
"""

import os
import sys
import unittest

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """Simple test cases for VectorStore functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create unique collection name for each test to avoid dimension conflicts
        import uuid
        test_collection_name = f"test_{uuid.uuid4().hex[:8]}"

        try:
            self.vector_store = VectorStore(collection_name=test_collection_name)
        except Exception as e:
            self.skipTest(f"Could not connect to ChromaDB Cloud: {str(e)}")

    def tearDown(self):
        """Clean up after test"""
        try:
            if hasattr(self, 'vector_store'):
                self.vector_store.clear_collection()
        except:
            pass

    def test_connection(self):
        """Test basic connection to ChromaDB Cloud"""
        healthy, message = self.vector_store.health_check()
        self.assertTrue(healthy)

    def test_collection_info(self):
        """Test getting collection information"""
        info = self.vector_store.get_collection_info()
        self.assertIn("collection_name", info)
        self.assertIn("total_documents", info)

    def test_add_and_search(self):
        """Test adding documents and searching"""
        # Test data
        chunks = ["This is a test document about AI."]
        metadata = [{"category": "test", "topic": "AI"}]
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]

        # Add documents
        success, message, ids = self.vector_store.add_documents(chunks, metadata, embeddings)
        self.assertTrue(success)
        self.assertEqual(len(ids), 1)

        # Search
        success, results, error = self.vector_store.similarity_search([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertTrue(success)
        self.assertGreater(len(results), 0)

    def test_delete_document(self):
        """Test deleting a document"""
        # Add a document first
        chunks = ["Document to delete"]
        metadata = [{"category": "test"}]
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]  # Use 5 dimensions like other tests

        success, _, ids = self.vector_store.add_documents(chunks, metadata, embeddings)
        self.assertTrue(success)

        # Delete it
        success, message = self.vector_store.delete_document(ids[0])
        self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()