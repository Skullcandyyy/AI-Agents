import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

# Custom styling
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 AI Chatbot")
st.write("Choose your AI mode and start chatting (type '0' to exit)")

# Initialize model
model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.9
)

# Mode selection
mode_option = st.selectbox(
    "Choose your AI mode:",
    ("Angry 😡", "Funny 😂", "Sad 😢")
)

# Map modes
if mode_option == "Angry 😡":
    mode = "you are an angry AI agent. you respond aggressively and impatiently"
elif mode_option == "Funny 😂":
    mode = "you are a funny AI agent. you respond with humor and jokes"
elif mode_option == "Sad 😢":
    mode = "you are a sad AI agent. you respond with empathy and understanding"

# Initialize session messages
if "messages" not in st.session_state or st.session_state.get("mode") != mode:
    st.session_state.messages = [SystemMessage(content=mode)]
    st.session_state.mode = mode

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# Input box
user_input = st.chat_input("Type your message...")

if user_input:
    if user_input == "0":
        st.stop()

    # Add user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    # Get response
    response = model.invoke(st.session_state.messages)

    # Add AI response
    st.session_state.messages.append(AIMessage(content=response.content))
    st.chat_message("assistant").write(response.content)