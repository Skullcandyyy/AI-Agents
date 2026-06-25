from langchain_huggingface import  HuggingFaceEndpoint

llm =HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",



)
import os
from huggingface_hub import login
login(os.getenv("HF_TOKEN"))

response = llm.invoke("What is the capital of France?")
print(response)