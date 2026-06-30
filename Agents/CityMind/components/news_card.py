import streamlit as st
import json

def render_news_card(json_str: str):
    try:
        data = json.loads(json_str)
        city = data.get("city", "Unknown")
        results = data.get("results", [])
        
        st.markdown(f"#### 📰 Latest News in {city.title()}")
        
        for article in results:
            title = article.get("title", "No Title")
            url = article.get("url", "#")
            content = article.get("content", "No content available")
            
            st.html(f'''
            <div class="premium-news-card">
                <a href="{url}" target="_blank" class="news-title">{title}</a>
                <div class="news-snippet">{content[:150]}...</div>
                <div class="news-footer">
                    <span>Source: Tavily Live Search</span>
                    <a href="{url}" target="_blank" style="color:#38BDF8;text-decoration:none;">Read More ↗</a>
                </div>
            </div>
            ''')
    except Exception as e:
        st.error(f"Failed to render news card: {str(e)}")
