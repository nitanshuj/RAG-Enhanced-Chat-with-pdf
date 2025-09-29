# RAG-Enhanced PDF Chat - Usage Guide

## Phase 4.2 Complete - Streamlit Integration

This guide shows how to use the fully integrated RAG-enhanced PDF chatbot.

## Quick Start

1. **Launch the application**:
   ```bash
   uv run streamlit run main.py
   ```

2. **Upload a PDF document**:
   - Use the file uploader in the sidebar
   - Select the appropriate document category
   - Click "Process Document"

3. **Start chatting**:
   - Ask questions about your document
   - Get AI-powered responses with source citations
   - Enable debug mode to see retrieval details

## Features Available

### Document Processing
- **PDF text extraction** with category-aware chunking
- **Automatic embedding generation** using AI/ML API
- **ChromaDB Cloud storage** for persistent vector search
- **Progress indicators** during processing

### RAG-Powered Chat
- **Intelligent document retrieval** using vector similarity
- **Category-filtered search** for targeted responses
- **Conversation history** with context awareness
- **Source attribution** showing relevant document sections

### Advanced Features
- **Debug mode** - Shows retrieval details and similarity scores
- **Session management** - Tracks queries and conversation state
- **Error handling** - Graceful failures with helpful messages
- **Real-time updates** - Live session statistics

## Interface Overview

### Sidebar
- **File Upload**: Choose PDF files to process
- **Category Selection**: Choose from 6 document types
- **Debug Toggle**: Enable detailed retrieval information
- **RAG Status**: Shows system state and query count
- **Processing Details**: Document analysis summary

### Main Chat Area
- **Document Info Bar**: Shows current document and chunk count
- **Chat Interface**: Ask questions and get responses
- **Source Citations**: Each response shows relevant document sections
- **Debug Information**: Detailed retrieval and processing data

## Document Categories

The system supports 6 specialized document types:

1. **Research Paper** - Academic papers with section-based analysis
2. **Article** - News articles and blog posts
3. **Book** - Book chapters and long-form content
4. **Receipts** - Financial documents with amount detection
5. **Terms & Conditions** - Legal documents with clause analysis
6. **Other** - General documents with standard processing

## Example Workflow

1. **Upload a research paper PDF**
2. **Select "Research Paper" category**
3. **Wait for processing** (embeddings + vector storage)
4. **Ask**: "What is the main hypothesis of this study?"
5. **Get response** with relevant sections and confidence scores
6. **Enable debug mode** to see retrieval details
7. **Continue conversation** with follow-up questions

## Response Format

Each AI response includes:
- **Main answer** based on retrieved document context
- **Source citations** with section names and relevance scores
- **Document statistics** showing chunks analyzed
- **Debug information** (when enabled) with technical details

## Debug Information

When debug mode is enabled, you'll see:
- **Query processing** details
- **Similarity search** results and scores
- **Context length** and formatting information
- **Source previews** with text snippets

## Troubleshooting

### Common Issues
- **"RAG system not initialized"** → Process a document first
- **"Failed to generate embedding"** → Check AI/ML API key
- **"Vector store initialization failed"** → Check ChromaDB credentials
- **"No relevant documents found"** → Try rephrasing your question

### Performance Tips
- Use specific questions for better retrieval
- Enable category filters for focused searches
- Check debug mode if responses seem off-topic
- Clear session and restart for fresh conversations

## Technical Details

### Architecture
- **Streamlit** frontend with responsive chat interface
- **SimpleRAGAgent** orchestrating the complete pipeline
- **ChromaDB Cloud** for scalable vector storage
- **AI/ML API** for embeddings and response generation

### Data Flow
1. PDF → Document Processor → Text chunks
2. Chunks → LLM Client → Vector embeddings
3. Embeddings → Vector Store → Persistent storage
4. Query → RAG Agent → Similarity search → Response

The system is now fully integrated and ready for production use with all Phase 4.1 and 4.2 requirements complete!