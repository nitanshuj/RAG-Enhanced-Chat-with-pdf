import streamlit as st
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.agent import SimpleRAGAgent
from src.utils import auto_detect_category, get_category_confidence, DOCUMENT_CATEGORIES
import src.llm_client as llm_client
import src.embeddings as embeddings

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
    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = False

def main():
    st.set_page_config(
        page_title="PDF Document Q&A Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better UI with dark mode support
    st.markdown("""
        <style>
        /* Main header styling */
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        .main-header p {
            color: #e0e7ff;
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        }

        /* Chat message styling - dark mode friendly */
        .stChatMessage {
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        /* Info boxes */
        .stAlert {
            border-radius: 8px;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: transform 0.2s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        /* File uploader - dark mode friendly */
        [data-testid="stFileUploader"] {
            border-radius: 10px;
            padding: 1rem;
            border: 2px dashed #667eea;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            color: #667eea;
            font-weight: 700;
        }

        /* Document info card */
        .doc-info-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* Info card - dark mode friendly */
        .info-card {
            padding: 10px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }

        /* Light mode specific */
        @media (prefers-color-scheme: light) {
            .info-card {
                background-color: #e7f3ff;
                color: #1a1a1a;
            }
            .welcome-card {
                background-color: #f8f9fa;
            }
            .feature-card {
                background-color: white;
                color: #1a1a1a;
            }
            .status-card-active {
                background-color: #d4edda;
                color: #155724;
            }
            .status-card-inactive {
                background-color: #fff3cd;
                color: #856404;
            }
        }

        /* Dark mode specific */
        @media (prefers-color-scheme: dark) {
            .info-card {
                background-color: rgba(102, 126, 234, 0.15);
                color: #e0e7ff;
            }
            .welcome-card {
                background-color: rgba(255, 255, 255, 0.05);
            }
            .feature-card {
                background-color: rgba(255, 255, 255, 0.08);
                color: #e0e0e0;
            }
            .status-card-active {
                background-color: rgba(40, 167, 69, 0.15);
                color: #7ed699;
            }
            .status-card-inactive {
                background-color: rgba(255, 193, 7, 0.15);
                color: #ffd966;
            }
        }

        /* Expander headers */
        .streamlit-expanderHeader {
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    # Sidebar - File upload and category selector
    with st.sidebar:
        st.markdown("### 📁 Document Upload")
        st.markdown("---")

        # File upload widget
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf']
        )

        # Document category selector with auto-detection
        if uploaded_file:
            # Auto-detect category from first portion of the document
            if 'auto_detected_category' not in st.session_state or st.session_state.uploaded_file != uploaded_file:
                # Extract first 2000 chars for detection
                try:
                    from src.document_processor import DocumentProcessor
                    temp_processor = DocumentProcessor()
                    sample_text = temp_processor.extract_text(uploaded_file.getbuffer())[:2000]
                    detected_cat = auto_detect_category(sample_text)
                    _, confidence = get_category_confidence(sample_text, detected_cat)

                    st.session_state.auto_detected_category = detected_cat
                    st.session_state.detection_confidence = confidence
                except Exception as e:
                    st.session_state.auto_detected_category = "Other"
                    st.session_state.detection_confidence = 0.3

            # Show auto-detection result
            detected = st.session_state.get('auto_detected_category', 'Other')
            confidence = st.session_state.get('detection_confidence', 0.5)

            # Find index of detected category for pre-selection
            default_index = DOCUMENT_CATEGORIES.index(detected) if detected in DOCUMENT_CATEGORIES else 3

            st.markdown(f"""
                <div class='info-card' style='border-left-color: #2196F3;'>
                    <strong>🔍 Auto-detected:</strong> {detected}<br>
                    <small>Confidence: {confidence:.0%}</small>
                </div>
            """, unsafe_allow_html=True)
            st.write("")

            selected_category = st.selectbox(
                "📋 Confirm or change category:",
                options=DOCUMENT_CATEGORIES,
                index=default_index,
                help="Auto-detected category is pre-selected. Change if needed."
            )

            if st.button("🚀 Process Document", use_container_width=True):
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
                            # Prepare documents for Langchain vector store
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

                            # Store in vector store (Langchain handles embeddings automatically)
                            with st.spinner("Generating embeddings and storing documents..."):
                                store_success, store_message, chunk_ids = st.session_state.vector_store.add_documents(
                                    chunks, vector_metadata, None
                                )

                            if store_success:
                                st.session_state.uploaded_file = uploaded_file
                                st.session_state.selected_category = selected_category
                                st.session_state.processed_data = result
                                st.session_state.document_processed = True

                                st.success("✅ Document processed and stored successfully!")
                                st.info(f"📊 Extracted {result['chunk_count']} text chunks")
                                st.info(f"💾 {store_message}")
                            else:
                                st.error(f"Failed to store document: {store_message}")
                        else:
                            st.error(f"Processing failed: {result['error']}")
                            st.session_state.document_processed = False

                    except Exception as e:
                        st.error(f"Error setting up RAG system: {str(e)}")
                        st.session_state.document_processed = False

        # Debug mode toggle
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        st.session_state.show_debug = st.checkbox("🐛 Show debug information", value=False)

        # RAG system status
        st.markdown("---")
        if st.session_state.rag_agent is not None:
            st.markdown("""
                <div class='status-card-active' style='padding: 10px; border-radius: 8px; border-left: 4px solid #28a745;'>
                    <strong>✅ RAG System Active</strong>
                </div>
            """, unsafe_allow_html=True)
            session_info = st.session_state.rag_agent.get_session_info()
            st.metric("💬 Total Queries", session_info['total_queries'])
        else:
            st.markdown("""
                <div class='status-card-inactive' style='padding: 10px; border-radius: 8px; border-left: 4px solid #ffc107;'>
                    <strong>⏳ RAG System Inactive</strong>
                </div>
            """, unsafe_allow_html=True)

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
                st.write("**Research Paper Analysis:**")
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
    st.markdown("""
        <div class="main-header">
            <h1>🤖 PDF Document Q&A Chatbot</h1>
            <p>Upload a PDF and ask questions powered by AI</p>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.document_processed and st.session_state.processed_data:
        # Document info bar with gradient background
        st.markdown(f"""
            <div class="doc-info-card">
                <h3 style='margin: 0 0 0.5rem 0;'>📄 {st.session_state.uploaded_file.name}</h3>
                <p style='margin: 0; opacity: 0.9;'>Category: {st.session_state.selected_category} |
                Chunks: {st.session_state.processed_data['chunk_count']} |
                Characters: {st.session_state.processed_data['metadata']['text_length']:,}</p>
            </div>
        """, unsafe_allow_html=True)

        # Clear button with document deletion and confirmation
        if not st.session_state.confirm_delete:
            # Show clear button in the right corner
            col1, col2, col3 = st.columns([6, 1, 1])
            with col3:
                if st.button("🗑️ Clear", use_container_width=True, help="Clear session and delete document from cloud storage"):
                    st.session_state.confirm_delete = True
                    st.rerun()
        else:
            # Show confirmation dialog centered with proper spacing
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div style='text-align: center; padding: 1rem; background-color: rgba(255, 193, 7, 0.1); border-radius: 8px; border: 2px solid #ffc107;'>
                    <h4 style='margin: 0; color: #ffc107;'>⚠️ Delete Document from Cloud?</h4>
                    <p style='margin: 0.5rem 0 0 0; opacity: 0.8;'>This will permanently remove all document chunks from ChromaDB Cloud storage.</p>
                </div>
            """, unsafe_allow_html=True)

            # Confirmation buttons with better layout
            col1, col2, col3, col4, col5 = st.columns([2.5, 1.5, 0.5, 1.5, 2.5])
            with col2:
                if st.button("✅ Yes, Delete", use_container_width=True, type="primary", key="confirm_yes"):
                    # Perform deletion
                    with st.spinner("Deleting from cloud storage..."):
                        # Delete from ChromaDB Cloud
                        if st.session_state.vector_store and st.session_state.uploaded_file:
                            document_filename = st.session_state.uploaded_file.name

                            # Delete all chunks for this document
                            success, message, count = st.session_state.vector_store.delete_documents_by_metadata(
                                {"document_id": document_filename}
                            )

                            if success:
                                if count > 0:
                                    st.success(f"✅ Deleted {count} chunks from cloud")
                                else:
                                    st.info("ℹ️ No documents found in cloud")
                            else:
                                st.error(f"❌ Failed: {message}")

                    # Clear session state
                    for key in ['uploaded_file', 'selected_category', 'processed_data', 'document_processed', 'messages', 'confirm_delete']:
                        if key in st.session_state:
                            del st.session_state[key]

                    # Small delay to show the message
                    import time
                    time.sleep(1.5)
                    st.rerun()

            with col4:
                if st.button("❌ Cancel", use_container_width=True, key="confirm_no"):
                    st.session_state.confirm_delete = False
                    st.rerun()

        st.markdown("---")

        # Display chat messages with custom styling
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
                    st.markdown(message["content"])

        # Chat input with custom placeholder
        if prompt := st.chat_input("💭 Ask a question about your document...", key="chat_input"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Generate response using RAG agent with streaming
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                full_response = ""

                # Stream the response
                for token in st.session_state.rag_agent.process_query_stream(
                    query=prompt,
                    category_filter=st.session_state.selected_category,
                    top_k=5
                ):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                # Final response without cursor
                response_placeholder.markdown(full_response)

                # Show debug information if enabled (non-streaming for now)
                # if st.session_state.show_debug:
                #     with st.expander("🔧 Debug Information"):
                #         st.write("Debug info available in non-streaming mode")

            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        # Welcome screen when no document is uploaded
        st.markdown("""
            <div class='welcome-card' style='text-align: center; padding: 3rem 2rem; border-radius: 10px; margin-top: 2rem;'>
                <h2 style='color: #667eea; margin-bottom: 1rem;'>👋 Welcome!</h2>
                <p style='font-size: 1.2rem; margin-bottom: 2rem; opacity: 0.8;'>
                    Get started by uploading a PDF document from the sidebar
                </p>
                <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;'>
                    <div class='feature-card' style='padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 200px;'>
                        <div style='font-size: 2rem; margin-bottom: 0.5rem;'>📄</div>
                        <strong>Upload PDF</strong>
                        <p style='font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.7;'>Support for various document types</p>
                    </div>
                    <div class='feature-card' style='padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 200px;'>
                        <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🤖</div>
                        <strong>AI Analysis</strong>
                        <p style='font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.7;'>Powered by advanced LLMs</p>
                    </div>
                    <div class='feature-card' style='padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 200px;'>
                        <div style='font-size: 2rem; margin-bottom: 0.5rem;'>💬</div>
                        <strong>Ask Questions</strong>
                        <p style='font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.7;'>Get instant answers from your docs</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

def generate_rag_response(prompt: str, category_filter: str) -> tuple:
    """Generate response using RAG agent"""
    try:
        if st.session_state.rag_agent is None:
            return "RAG system not initialized. Please process a document first.", None

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
            return f"Error generating response: {error}", {"error": error}

    except Exception as e:
        return f"RAG system error: {str(e)}", {"exception": str(e)}

if __name__ == "__main__":
    main()