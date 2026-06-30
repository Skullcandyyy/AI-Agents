import streamlit as st
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Must be the first Streamlit command
st.set_page_config(
    page_title="CityMind AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local imports
from components.utils import init_session_state, get_llm_with_tools
from components.bg_animation import render_bg_animation
from components.sidebar import render_sidebar
from components.hero import render_hero
from components.chat import render_chat_history, handle_chat_input
from components.approval_modal import render_approval_modal
from components.analytics import render_analytics
from components.footer import render_footer

def load_css():
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("styles.css not found.")

def main():
    # Load custom CSS
    load_css()
    
    # Render 3D Background
    render_bg_animation()
    
    init_session_state()
    
    # Render Modals
    if st.session_state.get("pending_tool_calls"):
        render_approval_modal()
        
    render_sidebar()
    
    # Layout
    # Left empty space, middle chat, right analytics
    col_left, col_main, col_right = st.columns([1, 6, 2])
    
    with col_main:
        render_hero()
        render_chat_history()
        
        # Get LLM with bound tools
        llm_with_tools = get_llm_with_tools()
        
        # Handle chat input at the bottom
        handle_chat_input(llm_with_tools)
        
    with col_right:
        render_analytics()
        
    render_footer()

if __name__ == "__main__":
    main()
