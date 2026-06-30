import streamlit as st

def render_footer():
    st.html('''
    <div style="text-align: center; padding: 20px 0; margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.1); color: #64748B; font-size: 0.9rem;">
        &copy; 2026 CityMind AI. Built with Streamlit & Langchain.<br>
        <span style="opacity: 0.7;">Powered by Mistral AI • OpenWeather • Tavily</span>
    </div>
    ''')
