import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional


class MovieInfo(BaseModel):
    movie_name: str
    release_year: int
    genre: List[str]
    director: str
    main_cast: List[str]
    main_characters: List[str]
    setting: str
    key_themes: List[str]
    scientific_technical_concepts: Optional[List[str]]
    summary: str


# Load env
load_dotenv()

# Model
model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.9
)

# Prompt
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
("human",
"""
Extract the movie information from the following paragraph:

{paragraph}
""")
])

# UI
st.set_page_config(page_title="Movie Info Extractor", page_icon="🎬")

st.title("🎬 Movie Information Extractor")

para = st.text_area("Enter your paragraph:")

if st.button("Extract Information"):
    if para:
        final_prompt = prompt.invoke({"paragraph": para})
        response = model.invoke(final_prompt)

        st.subheader("Extracted Information:")
        st.write(response.content)
    else:
        st.warning("Please enter a paragraph.")