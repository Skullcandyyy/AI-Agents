from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI

model= ChatMistralAI(model="mistral-small", temperature=0.9, max_tokens=20)

response=model.invoke("give me a paragraph on machine learning" )
print(response.content)