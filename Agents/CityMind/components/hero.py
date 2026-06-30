import streamlit as st
import textwrap

def render_hero():
    # Only render if no user messages
    user_msgs = [m for m in st.session_state.messages if m.type == "human"]
    if not user_msgs:
        st.markdown(textwrap.dedent('''
        <div class="hero-container">
            <div style="display: flex; justify-content: center; align-items: center; gap: 15px;">
                <svg width="72" height="72" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect width="72" height="72" rx="16" fill="url(#bg-grad)"/>
                    <defs>
                        <linearGradient id="bg-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#4F8CFF" />
                            <stop offset="100%" stop-color="#7B61FF" />
                        </linearGradient>
                    </defs>
                    <line x1="36" y1="20" x2="36" y2="10" stroke="#CBD5E1" stroke-width="4" stroke-linecap="round"/>
                    <circle cx="36" cy="10" r="4" fill="#FDE047" />
                    <rect x="14" y="20" width="44" height="38" rx="12" fill="#F8FAFC" stroke="#94A3B8" stroke-width="2"/>
                    <rect x="8" y="32" width="6" height="14" rx="3" fill="#94A3B8"/>
                    <rect x="58" y="32" width="6" height="14" rx="3" fill="#94A3B8"/>
                    <circle cx="26" cy="34" r="5" fill="#0F172A"/>
                    <circle cx="27" cy="33" r="1.5" fill="#FFFFFF"/>
                    <circle cx="46" cy="34" r="5" fill="#0F172A"/>
                    <circle cx="47" cy="33" r="1.5" fill="#FFFFFF"/>
                    <circle cx="20" cy="40" r="3" fill="#F472B6" opacity="0.6"/>
                    <circle cx="52" cy="40" r="3" fill="#F472B6" opacity="0.6"/>
                    <path d="M 32 44 Q 36 48 40 44" stroke="#0F172A" stroke-width="2.5" stroke-linecap="round" fill="none"/>
                </svg>
                <h1 class="hero-title" style="margin-bottom:0;"><span>CityMind AI</span></h1>
            </div>
            <p class="hero-subtitle">Real-time Weather • Live News • AI Agent • Tool Calling</p>
        </div>
        '''), unsafe_allow_html=True)
        
        # Action Grid using Streamlit buttons
        st.markdown("<div style='max-width: 600px; margin: 0 auto 40px auto;'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🌤️ Weather", use_container_width=True):
                st.session_state.prefill_prompt = "What is the current weather in popular cities of India like Mumbai, Delhi, and Bangalore?"
                st.rerun()
                
        with col2:
            if st.button("📰 Live News", use_container_width=True):
                st.session_state.prefill_prompt = "What is the latest news in popular cities of India like Mumbai, Delhi, and Bangalore?"
                st.rerun()
                
        st.markdown("</div>", unsafe_allow_html=True)
