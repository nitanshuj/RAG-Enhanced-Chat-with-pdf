import streamlit as st
import os
import sys
import time

# Set up paths so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.agent import SimpleRAGAgent
import src.llm_client as llm_client
from src.config import DOCUMENT_CATEGORIES

# Set page configuration with a modern dark theme mindset
st.set_page_config(
    page_title="CognitivePDF - Advanced RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS
st.markdown("""
<style>
    /* Main Background and Colors */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f6fc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 700;
    }
    
    /* Gradient Title */
    .gradient-text {
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Card Container */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: #58a6ff;
        box-shadow: 0 4px 20px rgba(88, 166, 255, 0.1);
    }
    
    /* Uploaded document list */
    .doc-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #21262d;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.5rem;
    }
    .doc-name {
        font-weight: 500;
        color: #c9d1d9;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 180px;
    }
    .doc-category {
        font-size: 0.75rem;
        background-color: #388bfd26;
        color: #58a6ff;
        border: 1px solid #388bfd4d;
        border-radius: 12px;
        padding: 0.1rem 0.5rem;
    }

    /* Streamlit Chat custom bubble overrides */
    div[data-testid="stChatMessage"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get or create components
@st.cache_resource
def get_backend_components():
    try:
        processor = DocumentProcessor()
        vector_store = VectorStore(collection_name="fastapi_docs")
        rag_agent = SimpleRAGAgent(llm_client, vector_store)
        return processor, vector_store, rag_agent, None
    except Exception as e:
        return None, None, None, str(e)

processor, vector_store, rag_agent, init_error = get_backend_components()

if init_error:
    st.error(f"Failed to initialize backend components: {init_error}")
    st.info("Please verify that your `.env` file is set up correctly with Chroma and API credentials.")
    st.stop()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fetch the unique documents stored in Chroma
def get_indexed_documents():
    try:
        collection = vector_store.vectorstore._collection
        results = collection.get(include=["metadatas"])
        if not results or not results.get('metadatas'):
            return []
        
        seen_docs = {}
        for meta in results['metadatas']:
            doc_id = meta.get('document_id')
            category = meta.get('category', 'Other')
            if doc_id:
                seen_docs[doc_id] = category
                
        return [{"filename": name, "category": cat} for name, cat in seen_docs.items()]
    except Exception as e:
        st.sidebar.error(f"Error reading indexed documents: {e}")
        return []

# Sidebar Content
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #f0f6fc;'>⚙️ Control Center</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Section 1: Upload Documents
    st.markdown("### 📤 Upload New Document")
    uploaded_file = st.file_uploader("Select a PDF", type=["pdf"])
    selected_category = st.selectbox("Document Category", DOCUMENT_CATEGORIES)
    
    if st.button("Process & Index Document", use_container_width=True, type="primary"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF and storing embeddings..."):
                try:
                    content = uploaded_file.read()
                    result = processor.process_document(content, selected_category, uploaded_file.name)
                    
                    if result.get('success'):
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
                        
                        # Add to vector store
                        store_success, store_message, ids = vector_store.add_documents(
                            chunks, vector_metadata, None
                        )
                        
                        if store_success:
                            st.success(f"Success! Indexed {len(ids)} chunks from '{uploaded_file.name}'.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"Store failed: {store_message}")
                    else:
                        st.error(f"Processing failed: {result.get('error')}")
                except Exception as e:
                    st.error(f"Error uploading document: {e}")
        else:
            st.warning("Please upload a PDF file first.")
            
    st.markdown("---")
    
    # Section 2: Indexed Documents List
    st.markdown("### 📚 Indexed Documents")
    docs = get_indexed_documents()
    if not docs:
        st.info("No documents currently indexed.")
    else:
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"""
                <div class="doc-item">
                    <span class="doc-name" title="{doc['filename']}">{doc['filename']}</span>
                    <span class="doc-category">{doc['category']}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                # Use a unique key for each delete button
                if st.button("🗑️", key=f"del_{doc['filename']}", help=f"Delete {doc['filename']}"):
                    with st.spinner(f"Deleting {doc['filename']}..."):
                        success, message, count = vector_store.delete_documents_by_metadata(
                            {"document_id": doc['filename']}
                        )
                        if success:
                            st.toast(f"Deleted {count} chunks for {doc['filename']}", icon="✅")
                            rag_agent.clear_session()
                            st.session_state.messages = []
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(message)

    st.markdown("---")
    
    # Section 3: Reset / Settings
    st.markdown("### 🧹 Database & Session")
    col_clear_chat, col_clear_db = st.columns(2)
    with col_clear_chat:
        if st.button("Clear Chat", use_container_width=True):
            rag_agent.clear_session()
            st.session_state.messages = []
            st.toast("Chat history cleared")
            st.rerun()
    with col_clear_db:
        if st.button("Reset DB", use_container_width=True, type="secondary", help="Clears entire collection"):
            try:
                vector_store.vectorstore.delete_collection()
                # Reinitialize
                vector_store = VectorStore(collection_name="fastapi_docs")
                rag_agent = SimpleRAGAgent(llm_client, vector_store)
                st.session_state.messages = []
                st.toast("Database reset completed")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting database: {e}")

# Main Area Header
st.markdown("<div class='gradient-text'>🧠 CognitivePDF</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Multi-Category Agentic RAG Q&A Hub</div>", unsafe_allow_html=True)

# Main Section - Chat filters and Stats
col_filter, col_stats = st.columns([2, 1])

with col_filter:
    category_filter = st.selectbox(
        "🎯 Search Category Filter",
        ["All Categories"] + DOCUMENT_CATEGORIES,
        index=0,
        help="Restrict the search and context to a single document category."
    )
    actual_filter = None if category_filter == "All Categories" else category_filter

with col_stats:
    session_info = rag_agent.get_session_info()
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.85rem; color: #8b949e; margin-bottom: 0.25rem;">RAG Engine Status</div>
        <div style="font-size: 1.25rem; font-weight: bold; color: #39d353;">● Operational</div>
        <div style="font-size: 0.75rem; color: #8b949e; margin-top: 0.25rem;">Total Queries Processed: {session_info.get('total_queries', 0)}</div>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything about the indexed documents..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream response
        try:
            for token in rag_agent.process_query_stream(
                query=prompt,
                category_filter=actual_filter,
                top_k=5
            ):
                full_response += token
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error streaming response: {e}")
            full_response = f"Error: {e}"
            message_placeholder.markdown(full_response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
