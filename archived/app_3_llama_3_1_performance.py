import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# from langchain_community.embeddings import HuggingFaceEmbeddings

# Large Language Model
MODEL = "llama3.1:8b"
model = Ollama(base_url="http://localhost:11434", model=MODEL)

# Inital Embeddings
# embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
embeddings = OllamaEmbeddings(model=MODEL)

@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template =     """
        Answer the question as detailed as possible from the provided context. 
        Make sure to provide all the details.
        If the answer is not in the provided context, just say, 
        "Answer is not available in the provided context."
        Don't provide a wrong answer.
        Context: {context}
        Question: {question}
        Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_query(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    
    st.set_page_config(page_title="Chat with PDF(s)")
    
    st.header("Chat with PDF(s) using Llama 3.1")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdf_docs = st.file_uploader("Upload PDFs here", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks)
                st.success("PDFs processed successfully!")

    if st.session_state.vector_store is not None:
        user_question = st.text_input("Ask a question about your PDFs:")
        if st.button("Ask Question") and user_question:
            response = process_query(user_question, st.session_state.vector_store)
            st.session_state.conversation.append(("You", user_question))
            st.session_state.conversation.append(("Llama 3.1", response))

        for role, message in st.session_state.conversation:
            if role == "You":
                st.write(f"You: {message}")
            else:
                st.write(f"Llama 3.1: {message}")


if __name__ == "__main__":
    main()            