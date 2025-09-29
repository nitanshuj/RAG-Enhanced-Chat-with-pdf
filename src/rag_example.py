"""
Example usage of the SimpleRAGAgent for testing and demonstration.

This script shows how to set up and use the RAG agent with sample documents.
"""

import os
import sys
from typing import List, Dict

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from src.vector_store import VectorStore
from src.agent import SimpleRAGAgent
import src.llm_client as llm_client
import src.embeddings as embeddings


def create_sample_documents() -> tuple:
    """
    Create sample documents for testing the RAG agent.

    Returns:
        Tuple of (chunks, metadata, embeddings)
    """
    chunks = [
        "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",

        "The scientific method involves making observations, forming hypotheses, conducting experiments, and analyzing results. This systematic approach helps researchers understand natural phenomena and develop new theories.",

        "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, particularly the burning of fossil fuels, have significantly contributed to recent climate change.",

        "A receipt is a document that serves as proof of purchase. It typically includes the date, items purchased, quantities, prices, taxes, and total amount paid. Receipts are important for financial record-keeping.",

        "Terms and conditions are legal agreements between service providers and users. They outline the rules, policies, and limitations that govern the use of a service or product. Users typically agree to these terms before accessing a service."
    ]

    metadata = [
        {
            "category": "Research Paper",
            "section": "introduction",
            "document_id": "ai_paper_001",
            "title": "Introduction to AI and Machine Learning"
        },
        {
            "category": "Research Paper",
            "section": "methodology",
            "document_id": "science_paper_001",
            "title": "Scientific Method in Research"
        },
        {
            "category": "Article",
            "section": "overview",
            "document_id": "climate_article_001",
            "title": "Understanding Climate Change"
        },
        {
            "category": "Receipts",
            "section": "definition",
            "document_id": "receipt_guide_001",
            "title": "What is a Receipt?"
        },
        {
            "category": "Terms & Conditions",
            "section": "overview",
            "document_id": "legal_doc_001",
            "title": "Understanding Terms and Conditions"
        }
    ]

    return chunks, metadata


def setup_rag_system() -> SimpleRAGAgent:
    """
    Set up the complete RAG system with sample documents.

    Returns:
        Configured SimpleRAGAgent instance
    """
    print("Setting up RAG system...")

    # Create vector store
    vector_store = VectorStore(collection_name="rag_demo")
    print("Vector store created")

    # Create RAG agent
    rag_agent = SimpleRAGAgent(llm_client, vector_store)
    print("RAG agent created")

    # Get sample documents
    chunks, metadata = create_sample_documents()
    print(f"Created {len(chunks)} sample documents")

    # Generate embeddings for documents
    print("Generating embeddings...")
    embedding_list = []
    failed_embeddings = 0

    for i, chunk in enumerate(chunks):
        success, embedding, error = embeddings.generate_embeddings(chunk)
        if success:
            embedding_list.append(embedding)
        else:
            print(f"Failed to generate embedding for chunk {i+1}: {error}")
            embedding_list.append(None)
            failed_embeddings += 1

    if failed_embeddings > 0:
        print(f"Warning: {failed_embeddings} embeddings failed to generate")

    # Add documents to vector store
    success, message, chunk_ids = vector_store.add_documents(chunks, metadata, embedding_list)

    if success:
        print(f"Successfully added documents: {message}")
    else:
        print(f"Failed to add documents: {message}")
        return None

    return rag_agent


def demo_rag_queries(rag_agent: SimpleRAGAgent):
    """
    Demonstrate RAG agent with various types of queries.

    Args:
        rag_agent: Configured SimpleRAGAgent instance
    """
    print("\n" + "="*50)
    print("RAG AGENT DEMONSTRATION")
    print("="*50)

    # Test queries for different categories
    test_queries = [
        {
            "query": "What is artificial intelligence and machine learning?",
            "category": "Research Paper",
            "description": "AI/ML question with Research Paper filter"
        },
        {
            "query": "How does the scientific method work?",
            "category": None,
            "description": "General scientific question (no filter)"
        },
        {
            "query": "What causes climate change?",
            "category": "Article",
            "description": "Climate question with Article filter"
        },
        {
            "query": "What information is typically on a receipt?",
            "category": "Receipts",
            "description": "Receipt question with Receipts filter"
        },
        {
            "query": "What are terms and conditions?",
            "category": "Terms & Conditions",
            "description": "Legal question with Terms & Conditions filter"
        }
    ]

    for i, test in enumerate(test_queries, 1):
        print(f"\n{i}. {test['description']}")
        print("-" * 30)
        print(f"Query: {test['query']}")
        if test['category']:
            print(f"Category Filter: {test['category']}")

        # Process query
        success, response_data, error = rag_agent.process_query(
            query=test['query'],
            category_filter=test['category'],
            top_k=3
        )

        if success:
            print(f"Answer: {response_data['answer'][:200]}...")
            print(f"Documents found: {response_data['documents_found']}")
            print(f"Sources: {[doc['metadata']['title'] for doc in response_data['sources']]}")
        else:
            print(f"Error: {error}")

    # Show session information
    print(f"\n{'-'*50}")
    print("SESSION SUMMARY")
    print("-" * 50)
    session_info = rag_agent.get_session_info()
    print(f"Total queries processed: {session_info['total_queries']}")
    print(f"Categories queried: {session_info['categories_queried']}")
    print(f"Conversation length: {session_info['conversation_length']}")


def main():
    """Main demonstration function."""
    try:
        # Set up RAG system
        rag_agent = setup_rag_system()

        if rag_agent is None:
            print("Failed to set up RAG system")
            return

        # Run demonstration
        demo_rag_queries(rag_agent)

        # Clean up
        print(f"\n{'-'*50}")
        print("Cleaning up...")
        rag_agent.vector_store.clear_collection()
        print("Demo completed successfully!")

    except Exception as e:
        print(f"Demo failed: {str(e)}")


if __name__ == "__main__":
    main()