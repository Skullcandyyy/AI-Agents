from dotenv import load_dotenv
load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#1. Prompt template
prompt=ChatPromptTemplate.from_template(
    "Explain{topic} in the simple words"

)
#2. Model
model=ChatMistralAI(model="mistral-small-latest")

#3. output parsers
parser=StrOutputParser()

#step-by-step manual flow

chain= prompt | model | parser  
result=chain.invoke("Machine Learning")  
print(result)