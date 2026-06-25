import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# ----------------------------
# STREAMLIT PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="CineSage RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 CineSage RAG Chatbot")
st.markdown("Upload a PDF and ask questions from the document.")

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=2000,
        value=1000,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=200,
        step=50
    )

    k_value = st.slider(
        "Number of Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=4
    )

# ----------------------------
# LOAD EMBEDDING MODEL
# ----------------------------
@st.cache_resource

def load_embedding_model():
    return HuggingFaceEmbeddings()

embedding_model = load_embedding_model()

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload your PDF file",
    type=["pdf"]
)

# ----------------------------
# PROCESS PDF
# ----------------------------
if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        with st.spinner("Processing PDF and creating vector database..."):

            # Load PDF
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            chunks = text_splitter.split_documents(documents)

            # Create Chroma vector store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="chroma-db"
            )

            # Persist DB
            vectorstore.persist()

            # Save in session state
            st.session_state.vectorstore = vectorstore

        st.success("Vector database created successfully!")

# ----------------------------
# LOAD EXISTING VECTORSTORE
# ----------------------------
if "vectorstore" not in st.session_state:

    if os.path.exists("chroma-db"):
        st.session_state.vectorstore = Chroma(
            persist_directory="chroma-db",
            embedding_function=embedding_model
        )

# ----------------------------
# CHAT SECTION
# ----------------------------
if "vectorstore" in st.session_state:

    st.divider()
    st.subheader("💬 Ask Questions")

    query = st.text_input("Enter your question")

    if query:

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k_value,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        with st.spinner("Searching document and generating answer..."):

            # Retrieve documents
            docs = retriever.invoke(query)

            context = "\n".join([
                doc.page_content for doc in docs
            ])

            # Prompt Template
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context,
say: 'I could not find the answer in the document.'
"""
                    ),
                    (
                        "human",
                        """Context:
{context}

Question:
{question}
"""
                    )
                ]
            )

            # Load LLM
            llm = ChatMistralAI(model="mistral-small-latest")

            # Final Prompt
            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            # Generate response
            response = llm.invoke(final_prompt)

        # Display answer
        st.markdown("### 🤖 AI Response")
        st.write(response.content)

        # Show retrieved chunks
        with st.expander("📚 Retrieved Chunks"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"### Chunk {i}")
                st.write(doc.page_content)
                st.divider()

else:
    st.info("Upload a PDF and create the vector database to start chatting.")
