from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.9
)
prompt = ChatPromptTemplate.from_messages([
    ("system",
"""
You are an expert information extraction assistant.

Your task is to read the given paragraph and extract important movie-related information in a clean, readable format.

Extract and present the following details:

Movie Name:
Release Year:
Genre:
Director:
Main Cast:
Main Characters:
Setting (time/place):
Key Themes:
Scientific/Technical Concepts (if any):
Summary (2–3 lines):

Instructions:
- Only use information present in the paragraph.
- Do NOT add or assume missing details.
- If any information is not available, write "Not mentioned".
- Keep the summary short and clear (2–3 lines max).
- Format the output neatly with clear labels (no JSON, no code blocks).


"""
),
('human',
"""
Extract the movie information from the following paragraph:
{paragraph}
""")
]
)
para=input("Give your paragraph: ")
final_prompt=prompt.invoke(
    {"paragraph":para}
)
response = model.invoke(final_prompt)
print(response.content)