import streamlit as st
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.agent import SimpleRAGAgent
import src.llm_client as llm_client
import src.embeddings as embeddings

# Document categories from documentation
DOCUMENT_CATEGORIES = [
    "Research Paper",
    "Article",
    "Book",
    "Other",
    "Receipts",
    "Terms & Conditions"
]

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = None
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'rag_agent' not in st.session_state:
        st.session_state.rag_agent = None
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False

def main():
    st.set_page_config(
        page_title="PDF Document Q&A Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session_state()

    # Sidebar - File upload and category selector
    with st.sidebar:
        st.header("Document Upload")

        # File upload widget
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf']
        )

        # Document category selector
        if uploaded_file:
            selected_category = st.selectbox(
                "Select document category:",
                options=DOCUMENT_CATEGORIES
            )

            if st.button("Process Document"):
                with st.spinner("Processing PDF and setting up RAG system..."):
                    try:
                        # Initialize vector store and RAG agent if not already done
                        if st.session_state.vector_store is None:
                            st.session_state.vector_store = VectorStore(collection_name="streamlit_docs")
                            st.session_state.rag_agent = SimpleRAGAgent(llm_client, st.session_state.vector_store)

                        # Process the document
                        result = st.session_state.processor.process_document(
                            uploaded_file.getbuffer(),
                            selected_category,
                            uploaded_file.name
                        )

                        if result['success']:
                            # Generate embeddings and store in vector store
                            chunks = result['chunks']
                            chunk_metadata = result.get('chunk_metadata', [])

                            # Prepare metadata for vector store
                            vector_metadata = []
                            for i, chunk_meta in enumerate(chunk_metadata):
                                meta = {
                                    'category': selected_category,
                                    'document_id': uploaded_file.name,
                                    'section': chunk_meta.get('section', f'Section {i+1}'),
                                    'chunk_index': i,
                                    'chunk_type': chunk_meta.get('chunk_type', 'content'),
                                    'confidence': chunk_meta.get('confidence', 0.5)
                                }
                                vector_metadata.append(meta)

                            # Generate embeddings for all chunks
                            embedding_list = []
                            embedding_progress = st.progress(0)

                            for i, chunk in enumerate(chunks):
                                success_emb, embedding, error = embeddings.generate_embeddings(chunk)
                                if success_emb:
                                    embedding_list.append(embedding)
                                else:
                                    embedding_list.append(None)
                                    st.warning(f"Failed to generate embedding for chunk {i+1}")

                                embedding_progress.progress((i + 1) / len(chunks))

                            embedding_progress.empty()

                            # Store in vector store
                            store_success, store_message, chunk_ids = st.session_state.vector_store.add_documents(
                                chunks, vector_metadata, embedding_list
                            )

                            if store_success:
                                st.session_state.uploaded_file = uploaded_file
                                st.session_state.selected_category = selected_category
                                st.session_state.processed_data = result
                                st.session_state.document_processed = True

                                st.success(f"✅ Document processed and stored successfully!")
                                st.info(f"📊 Extracted {result['chunk_count']} text chunks")
                                st.info(f"🗄️ {store_message}")
                            else:
                                st.error(f"❌ Failed to store document: {store_message}")
                        else:
                            st.error(f"❌ Processing failed: {result['error']}")
                            st.session_state.document_processed = False

                    except Exception as e:
                        st.error(f"❌ Error setting up RAG system: {str(e)}")
                        st.session_state.document_processed = False

        # Debug mode toggle
        st.session_state.show_debug = st.checkbox("🔍 Show debug information", value=False)

        # RAG system status
        if st.session_state.rag_agent is not None:
            st.success("🤖 RAG System Active")
            session_info = st.session_state.rag_agent.get_session_info()
            st.caption(f"Queries: {session_info['total_queries']}")

    # Show processing details when document is processed
    if st.session_state.document_processed and st.session_state.processed_data:
        with st.sidebar.expander("📋 Processing Details"):
            result = st.session_state.processed_data
            selected_category = st.session_state.selected_category

            st.write(f"**Pages detected:** {result['metadata']['estimated_pages']}")
            st.write(f"**Text length:** {result['metadata']['text_length']:,} characters")
            st.write(f"**Chunking method:** {result['processing_info']['chunk_method']}")

            # Category-specific info
            if selected_category == "Research Paper":
                st.write("🔬 **Research Paper Analysis:**")
                sections_found = result['metadata'].get('sections_found', [])
                if sections_found:
                    st.write(f"**Sections identified:** {', '.join(sections_found)}")
                    st.write(f"**Section count:** {len(sections_found)}")
            elif selected_category == "Receipts":
                amounts = result['metadata'].get('detected_amounts', 0)
                st.write(f"**Financial amounts detected:** {amounts}")
            else:
                st.write(f"**Word count:** {result['metadata'].get('word_count', 'N/A')}")
                st.write(f"**Processing:** Standard chunking with overlap")

    # Main area - Chat interface
    st.header("PDF Document Q&A Chatbot")

    if st.session_state.document_processed and st.session_state.processed_data:
        # Document info bar
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info(f"📄 **{st.session_state.uploaded_file.name}** ({st.session_state.selected_category})")
        with col2:
            st.metric("Chunks", st.session_state.processed_data['chunk_count'])
        with col3:
            if st.button("🗑️ Clear"):
                for key in ['uploaded_file', 'selected_category', 'processed_data', 'document_processed', 'messages']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Generate response using RAG agent
            with st.chat_message("assistant"):
                with st.spinner("Searching documents and generating response..."):
                    response_text, debug_info = generate_rag_response(
                        prompt,
                        st.session_state.selected_category
                    )
                    st.write(response_text)

                    # Show debug information if enabled
                    if st.session_state.show_debug and debug_info:
                        with st.expander("🔍 Debug Information"):
                            st.json(debug_info)

            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response_text})
    else:
        st.warning("📤 Please upload and process a PDF document to start chatting.")

def generate_rag_response(prompt: str, category_filter: str) -> tuple:
    """Generate response using RAG agent"""
    try:
        if st.session_state.rag_agent is None:
            return "❌ RAG system not initialized. Please process a document first.", None

        # Process query through RAG agent
        success, response_data, error = st.session_state.rag_agent.process_query(
            query=prompt,
            category_filter=category_filter,
            top_k=5,
            include_history=True
        )

        if success:
            answer = response_data['answer']
            sources = response_data['sources']

            # Format response without sources
            formatted_response = answer

            # Prepare debug information
            debug_info = {
                "query_processed": response_data.get('query_processed', prompt),
                "category_filter": category_filter,
                "documents_found": response_data['documents_found'],
                "context_length": len(response_data.get('context_used', '')),
                "sources": [
                    {
                        "section": src.get('metadata', {}).get('section', 'Unknown'),
                        "similarity": src.get('similarity', 0),
                        "text_preview": src.get('text', '')[:100] + "..."
                    }
                    for src in sources[:3]
                ]
            }

            return formatted_response, debug_info

        else:
            return f"❌ Error generating response: {error}", {"error": error}

    except Exception as e:
        return f"❌ RAG system error: {str(e)}", {"exception": str(e)}

if __name__ == "__main__":
    main()