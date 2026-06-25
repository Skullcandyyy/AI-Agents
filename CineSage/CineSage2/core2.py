from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_community import ResponseSchema, StructuredOutputParser
movie_schema = ResponseSchema(
    name="movie",
    description="Information about a movie"
)

parser = StructuredOutputParser.from_response_schemas([movie_schema])

model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.9
)
prompt = ChatPromptTemplate.from_messages([
    ("system",
"""
Extract Movie Information from the paragraph
{format_instructions}

"""),
('human',"{paragraph}")
]
)

para=input("Give your paragraph: ")
final_prompt=prompt.invoke(
    {"paragraph":para, "format_instructions": parser.get_format_instructions()
}
)
response = model.invoke(final_prompt)
print(response.content)