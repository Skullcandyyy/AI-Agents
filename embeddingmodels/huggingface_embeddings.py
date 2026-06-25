from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
texts=[
    "Hello this is Rohit Kumar",
    "Hello your name is youtube",
    "and youb all are beautiful"]
vectors = embeddings.embed_documents(texts)
print(vectors)