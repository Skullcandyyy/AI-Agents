import streamlit as st
import textwrap

def render_sidebar():
    with st.sidebar:
        st.markdown(textwrap.dedent('''
        <div class="sidebar-header" style="display: flex; flex-direction: row; align-items: center; text-align: left; gap: 10px;">
            <svg width="48" height="48" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
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
            <div>
                <h1 style="margin-bottom: 5px; font-size: 1.5rem;">CityMind AI</h1>
                <p style="margin: 0;">Your Intelligent City Assistant</p>
            </div>
        </div>
        '''), unsafe_allow_html=True)

        st.markdown("### 🔌 Connected APIs")
        
        # API Status
        for api_name, is_connected in st.session_state.api_status.items():
            status_class = "status-dot" if is_connected else "status-dot offline"
            st.html(f'''
            <div class="status-indicator">
                <div class="{status_class}"></div>
                <span>{api_name}</span>
            </div>
            ''')
            
        st.markdown("---")
        st.markdown("### 🧰 Available Tools")
        st.markdown("- 🌤️ Live Weather (OpenWeather)")
        st.markdown("- 📰 Live News (Tavily)")
        
        st.markdown("---")
        st.markdown("### 💬 Actions")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [m for m in st.session_state.messages if m.type == "system"]
            st.session_state.stats["total_messages"] = 0
            st.rerun()
            
        # Social links
        st.html("<br><br>")
        st.html('''
        <div style="display:flex; justify-content: space-around; opacity: 0.7;">
            <a href="#" style="color:white; text-decoration:none;">GitHub</a> •
            <a href="#" style="color:white; text-decoration:none;">LinkedIn</a> •
            <a href="#" style="color:white; text-decoration:none;">Portfolio</a>
        </div>
        ''')
