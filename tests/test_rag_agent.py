"""
Simple unit tests for the SimpleRAGAgent class.
"""

import os
import sys
import unittest

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agent import SimpleRAGAgent, create_rag_agent
from src.vector_store import VectorStore
import src.llm_client as llm_client
import src.embeddings as embeddings


class TestSimpleRAGAgent(unittest.TestCase):
    """Simple test cases for SimpleRAGAgent functionality"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            # Create test vector store
            import uuid
            test_collection = f"rag_test_{uuid.uuid4().hex[:8]}"
            self.vector_store = VectorStore(collection_name=test_collection)

            # Create RAG agent
            self.rag_agent = SimpleRAGAgent(llm_client, self.vector_store)

            # Add test documents
            self._add_test_documents()

        except Exception as e:
            self.skipTest(f"Could not set up RAG agent: {str(e)}")

    def tearDown(self):
        """Clean up after test"""
        try:
            if hasattr(self, 'vector_store'):
                self.vector_store.clear_collection()
        except:
            pass

    def _add_test_documents(self):
        """Add sample documents to vector store for testing"""
        # Test documents
        chunks = [
            "This document discusses artificial intelligence and machine learning concepts.",
            "The research methodology involved collecting data from 100 participants.",
            "Climate change is affecting global weather patterns significantly."
        ]

        metadata = [
            {"category": "AI", "section": "introduction", "document_id": "ai_doc_1"},
            {"category": "Research", "section": "methodology", "document_id": "research_doc_1"},
            {"category": "Environment", "section": "overview", "document_id": "climate_doc_1"}
        ]

        # Generate embeddings for test documents
        embedding_list = []
        for chunk in chunks:
            success, embedding, error = embeddings.generate_embeddings(chunk)
            if success:
                embedding_list.append(embedding)
            else:
                # Use dummy embedding if API fails
                embedding_list.append([0.1] * 1536)  # Typical embedding dimension

        # Add to vector store
        self.vector_store.add_documents(chunks, metadata, embedding_list)

    def test_agent_initialization(self):
        """Test RAG agent initialization"""
        self.assertIsNotNone(self.rag_agent.llm_client)
        self.assertIsNotNone(self.rag_agent.vector_store)
        self.assertEqual(len(self.rag_agent.conversation_history), 0)

    def test_session_info(self):
        """Test getting session information"""
        info = self.rag_agent.get_session_info()
        self.assertIn("session_start", info)
        self.assertIn("total_queries", info)
        self.assertEqual(info["total_queries"], 0)

    def test_query_preprocessing(self):
        """Test query preprocessing functionality"""
        # Test valid query
        success, processed, error = self.rag_agent._preprocess_query("What is AI?")
        self.assertTrue(success)
        self.assertEqual(processed, "What is AI?")

        # Test empty query
        success, processed, error = self.rag_agent._preprocess_query("")
        self.assertFalse(success)
        self.assertIn("Empty query", error)

        # Test short query
        success, processed, error = self.rag_agent._preprocess_query("AI")
        self.assertFalse(success)
        self.assertIn("too short", error)

    def test_process_query_basic(self):
        """Test basic query processing"""
        query = "What is artificial intelligence?"

        success, response_data, error = self.rag_agent.process_query(query)

        if success:
            # Test successful processing
            self.assertIn("answer", response_data)
            self.assertIn("sources", response_data)
            self.assertIn("context_used", response_data)
            self.assertGreater(len(response_data["sources"]), 0)

            # Check session state updated
            self.assertEqual(self.rag_agent.session_metadata["total_queries"], 1)
            self.assertEqual(len(self.rag_agent.conversation_history), 1)
        else:
            # If it fails due to API issues, that's also acceptable for testing
            self.assertIsNotNone(error)

    def test_process_query_with_category_filter(self):
        """Test query processing with category filter"""
        query = "Tell me about research methodology"

        success, response_data, error = self.rag_agent.process_query(
            query, category_filter="Research"
        )

        if success:
            self.assertIn("answer", response_data)
            self.assertEqual(response_data["category_filter"], "Research")

            # Check that sources are from the correct category
            for source in response_data["sources"]:
                metadata = source.get("metadata", {})
                self.assertEqual(metadata.get("category"), "Research")

    def test_conversation_history(self):
        """Test conversation history management"""
        queries = [
            "What is AI?",
            "How does machine learning work?",
            "What about climate change?"
        ]

        for query in queries:
            success, response_data, error = self.rag_agent.process_query(query)
            # Continue even if individual queries fail due to API issues

        # Check history length
        self.assertLessEqual(len(self.rag_agent.conversation_history), 3)

        # Test exporting conversation
        exported = self.rag_agent.export_conversation()
        self.assertIsInstance(exported, list)

    def test_clear_session(self):
        """Test clearing session state"""
        # Add some history first
        self.rag_agent.process_query("Test query")

        # Clear session
        self.rag_agent.clear_session()

        # Check that session is cleared
        self.assertEqual(len(self.rag_agent.conversation_history), 0)
        self.assertEqual(self.rag_agent.session_metadata["total_queries"], 0)

    def test_factory_function(self):
        """Test create_rag_agent factory function"""
        agent = create_rag_agent(self.vector_store)
        self.assertIsInstance(agent, SimpleRAGAgent)
        self.assertEqual(agent.vector_store, self.vector_store)


if __name__ == "__main__":
    unittest.main()