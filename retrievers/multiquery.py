from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

docs=[
    Document(page_content="gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support vector machines are supervised learning algorithms."),
]
embeddings=HuggingFaceEmbeddings()
vectorstore= Chroma.from_documents(docs,embeddings)
retriever=vectorstore.as_retriever()
llm=ChatMistralAI(model="mistral-small-latest")
multi_query_retriever=MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)
query="What is gradient descent?"
docs=multi_query_retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"\nResult{i+1}")
    print(doc.page_content)
