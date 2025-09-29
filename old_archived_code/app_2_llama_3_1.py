import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.vectorstores import FAISS

MODEL = "llama3.1:8b"
model = Ollama(base_url="http://localhost:11434", model=MODEL)
embeddings_ = OllamaEmbeddings(model=MODEL)
embedding_path = "G:/My Drive/Study/LLM - Large Language Models/LLM-Chat-with-pdf-using-Llama2/embeddings"

@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# @st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

# @st.cache_resource
def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings_)
    vector_store.save_local("faiss_index")
    # documents = [Document(page_content=text) for text in text_chunks]
    # vector_store = Chroma.from_documents(documents=documents, 
    #                                      embedding=embeddings_, 
    #                                      persist_directory=embedding_path)
    # vector_store.persist()
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
    
    prompt = PromptTemplate(template=prompt_template, 
                            input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def user_input(user_question, vector_store):
    # docs = vector_store.similarity_search(user_question)
    new_db = FAISS.load_local("faiss_index", 
                              embeddings=embeddings_, 
                              allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents": docs,
            "question": user_question
        }
    )
    st.write("Reply: ", response['output_text'])

def streamlit_application():
    st.set_page_config("Chat with PDF(s)")
    st.header("Chat with PDF(s)")

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input("Ask a question for the PDF")

    if user_question and st.session_state.vector_store:
        user_input(user_question, st.session_state.vector_store)
    
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(label="Upload the PDF(s) (one or multiple)", 
                                    accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vector_store(text_chunks)
                st.success("Done!!")

if __name__ == "__main__":
    streamlit_application()


