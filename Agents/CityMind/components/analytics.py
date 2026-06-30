import streamlit as st

def render_analytics():
    st.markdown("### 📊 Session Dashboard")
    
    stats = st.session_state.stats
    
    st.html(f'''
    <div style="display: flex; flex-direction: column; gap: 15px;">
        <div class="stat-card">
            <div>
                <div style="color: #94A3B8; font-size: 0.9rem;">Total Messages</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #F8FAFC;">{stats['total_messages']}</div>
            </div>
            <div style="font-size: 2rem; opacity: 0.8;">💬</div>
        </div>
        
        <div class="stat-card">
            <div>
                <div style="color: #94A3B8; font-size: 0.9rem;">Weather Calls</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #4F8CFF;">{stats['weather_calls']}</div>
            </div>
            <div style="font-size: 2rem; opacity: 0.8;">🌤️</div>
        </div>
        
        <div class="stat-card">
            <div>
                <div style="color: #94A3B8; font-size: 0.9rem;">News Calls</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #7B61FF;">{stats['news_calls']}</div>
            </div>
            <div style="font-size: 2rem; opacity: 0.8;">📰</div>
        </div>
    </div>
    ''')
