"""
🌍 CityMind AI Agent — Premium Streamlit UI
"An Intelligent AI Agent that thinks, searches, and answers using real-world tools."
"""

# ═══════════════════════════════════════════════════════════════
# 1. IMPORTS & PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

import os
import time
import json
import datetime
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import streamlit.components.v1 as components
import requests
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tavily import TavilyClient

st.set_page_config(
    page_title="CityMind AI Agent",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# 2. TOOL DEFINITIONS (Unmodified from Backend)
# ═══════════════════════════════════════════════════════════════

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    API_KEY = os.getenv("OPEN_WEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if str(data.get("cod")) != "200":
        return f"Error : {data.get('message', 'Could not fetch weather data')}"
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"Weather in {city}:{temp}K,{desc}"


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def get_news(city: str) -> str:
    """Get latest news of a city"""
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="fast",
        max_results=5
    )
    results = response.get("results", [])
    if not results:
        return f"No news found for {city}."
    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        news_list.append(f"-{title}\n 🔗 {url}\n  📰 {snippet[:100]}...")
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


# ═══════════════════════════════════════════════════════════════
# 3. LLM SETUP (Unmodified from Backend)
# ═══════════════════════════════════════════════════════════════

llm = ChatMistralAI(model="mistral-large-latest")
TOOLS_MAP = {"get_weather": get_weather, "get_news": get_news}
TOOL_LIST = [get_weather, get_news]


# ═══════════════════════════════════════════════════════════════
# 4. CSS STYLES
# ═══════════════════════════════════════════════════════════════

def get_css(theme: str = "dark") -> str:
    if theme == "dark":
        bg = "#0B1120"
        bg_secondary = "#111827"
        text = "#F8FAFC"
        text_secondary = "#94A3B8"
        card_bg = "rgba(255,255,255,0.05)"
        card_border = "rgba(255,255,255,0.08)"
        card_hover_bg = "rgba(255,255,255,0.09)"
        input_bg = "rgba(255,255,255,0.06)"
        input_border = "rgba(124,58,237,0.3)"
        sidebar_bg = "rgba(11,17,32,0.95)"
        code_bg = "rgba(0,0,0,0.4)"
        scrollbar_thumb = "rgba(124,58,237,0.3)"
        user_bubble = "linear-gradient(135deg, #7C3AED, #6D28D9)"
        ai_bubble = "rgba(255,255,255,0.06)"
        shadow_color = "rgba(0,0,0,0.4)"
    else:
        bg = "#F1F5F9"
        bg_secondary = "#FFFFFF"
        text = "#0F172A"
        text_secondary = "#64748B"
        card_bg = "rgba(255,255,255,0.8)"
        card_border = "rgba(0,0,0,0.08)"
        card_hover_bg = "rgba(255,255,255,0.95)"
        input_bg = "rgba(255,255,255,0.9)"
        input_border = "rgba(124,58,237,0.25)"
        sidebar_bg = "rgba(241,245,249,0.97)"
        code_bg = "rgba(0,0,0,0.05)"
        scrollbar_thumb = "rgba(124,58,237,0.2)"
        user_bubble = "linear-gradient(135deg, #7C3AED, #6D28D9)"
        ai_bubble = "rgba(0,0,0,0.04)"
        shadow_color = "rgba(0,0,0,0.08)"

    # NOTE: No leading spaces on any line to avoid markdown code-block rendering
    return f"""<style>
.stApp {{
background-color: {bg};
color: {text};
}}
html, body, [class*="st-"] {{
color: {text} !important;
}}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
.stDeployButton {{ display: none !important; }}
header[data-testid="stHeader"] {{
background: rgba(0,0,0,0) !important;
backdrop-filter: none !important;
}}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
background: {scrollbar_thumb};
border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{ background: rgba(124,58,237,0.5); }}
section[data-testid="stSidebar"] {{
background: {sidebar_bg};
backdrop-filter: blur(30px);
-webkit-backdrop-filter: blur(30px);
border-right: 1px solid {card_border};
}}
section[data-testid="stSidebar"] * {{
color: {text} !important;
}}
.glass-card {{
background: {card_bg};
backdrop-filter: blur(20px);
-webkit-backdrop-filter: blur(20px);
border: 1px solid {card_border};
border-radius: 16px;
padding: 24px;
transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
position: relative;
overflow: hidden;
}}
.glass-card::before {{
content: '';
position: absolute;
top: 0; left: 0; right: 0;
height: 2px;
background: linear-gradient(90deg, #7C3AED, #06B6D4, #22C55E);
opacity: 0;
transition: opacity 0.35s ease;
}}
.glass-card:hover {{
background: {card_hover_bg};
border-color: rgba(124,58,237,0.35);
transform: translateY(-3px);
box-shadow: 0 16px 48px {shadow_color};
}}
.glass-card:hover::before {{ opacity: 1; }}
.gradient-title {{
font-size: 3.2rem;
font-weight: 800;
background: linear-gradient(270deg, #7C3AED, #06B6D4, #22C55E, #F59E0B, #7C3AED);
background-size: 400% 400%;
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
animation: gradient-shift 6s ease infinite;
line-height: 1.2;
}}
@keyframes gradient-shift {{
0% {{ background-position: 0% 50%; }}
50% {{ background-position: 100% 50%; }}
100% {{ background-position: 0% 50%; }}
}}
@keyframes fade-in-up {{
from {{ opacity: 0; transform: translateY(24px); }}
to {{ opacity: 1; transform: translateY(0); }}
}}
.fade-in {{
animation: fade-in-up 0.7s cubic-bezier(0.16,1,0.3,1) forwards;
opacity: 0;
}}
.fade-in-d1 {{ animation-delay: 0.1s; }}
.fade-in-d2 {{ animation-delay: 0.25s; }}
.fade-in-d3 {{ animation-delay: 0.4s; }}
.fade-in-d4 {{ animation-delay: 0.55s; }}
.particle {{
position: absolute;
border-radius: 50%;
pointer-events: none;
animation: particle-float 8s ease-in-out infinite;
}}
@keyframes particle-float {{
0%, 100% {{ transform: translateY(0px) translateX(0px) scale(1); }}
25% {{ transform: translateY(-25px) translateX(15px) scale(1.05); }}
50% {{ transform: translateY(-10px) translateX(-10px) scale(0.95); }}
75% {{ transform: translateY(-35px) translateX(8px) scale(1.02); }}
}}
.glow-purple {{
box-shadow: 0 0 40px rgba(124,58,237,0.15), 0 0 80px rgba(124,58,237,0.05);
}}
.glow-cyan {{
box-shadow: 0 0 40px rgba(6,182,212,0.15), 0 0 80px rgba(6,182,212,0.05);
}}
.stChatMessage[data-testid="stChatMessageAssistant"] .stChatMessageContent {{
background: {ai_bubble};
border-radius: 4px 16px 16px 16px;
border: 1px solid {card_border};
padding: 16px 20px;
}}
.stChatMessage[data-testid="stChatMessageUser"] .stChatMessageContent {{
background: {user_bubble};
border-radius: 16px 4px 16px 16px;
padding: 14px 20px;
}}
.stChatMessage[data-testid="stChatMessageUser"] .stChatMessageContent * {{
color: #FFFFFF !important;
}}
.stChatMessage {{
padding: 4px 0;
}}
.stChatInputContainer {{
border: 1px solid {input_border} !important;
border-radius: 16px !important;
background: {input_bg} !important;
backdrop-filter: blur(20px) !important;
transition: all 0.3s ease !important;
}}
.stChatInputContainer:focus-within {{
border-color: #7C3AED !important;
box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}}
.stChatInputTextArea {{
color: {text} !important;
background: transparent !important;
}}
.stChatInputTextArea::placeholder {{
color: {text_secondary} !important;
}}
.typing-indicator {{
display: flex;
align-items: center;
gap: 10px;
color: {text_secondary};
font-size: 0.9rem;
}}
.typing-dots {{
display: flex;
gap: 4px;
}}
.typing-dots span {{
width: 8px;
height: 8px;
border-radius: 50%;
background: #7C3AED;
animation: dot-bounce 1.4s infinite both;
}}
.typing-dots span:nth-child(1) {{ animation-delay: 0s; }}
.typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
.typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
@keyframes dot-bounce {{
0%, 80%, 100% {{ transform: scale(0.4); opacity: 0.4; }}
40% {{ transform: scale(1); opacity: 1; }}
}}
.tool-call-card {{
background: rgba(124,58,237,0.08);
border: 1px solid rgba(124,58,237,0.2);
border-radius: 12px;
padding: 14px 18px;
margin: 8px 0;
display: flex;
align-items: center;
gap: 10px;
font-size: 0.9rem;
animation: fade-in-up 0.4s ease forwards;
}}
.tool-call-card .tool-icon {{
font-size: 1.3rem;
}}
.approval-panel {{
background: {card_bg};
backdrop-filter: blur(24px);
border: 1px solid rgba(124,58,237,0.25);
border-radius: 20px;
padding: 32px;
animation: fade-in-up 0.5s ease forwards;
box-shadow: 0 0 60px rgba(124,58,237,0.1);
}}
.approval-btn {{
display: inline-flex;
align-items: center;
gap: 8px;
padding: 12px 28px;
border-radius: 12px;
font-weight: 600;
font-size: 1rem;
cursor: pointer;
border: none;
transition: all 0.3s ease;
}}
.approve-btn {{
background: linear-gradient(135deg, #22C55E, #16A34A);
color: white;
}}
.approve-btn:hover {{
transform: translateY(-2px);
box-shadow: 0 8px 24px rgba(34,197,94,0.3);
}}
.deny-btn {{
background: rgba(239,68,68,0.15);
color: #EF4444;
border: 1px solid rgba(239,68,68,0.3);
}}
.deny-btn:hover {{
background: rgba(239,68,68,0.25);
transform: translateY(-2px);
}}
.timeline-container {{
padding-left: 8px;
}}
.timeline-item {{
position: relative;
padding-left: 36px;
padding-bottom: 20px;
border-left: 2px solid rgba(124,58,237,0.2);
animation: fade-in-up 0.4s ease forwards;
}}
.timeline-item:last-child {{
border-left-color: transparent;
padding-bottom: 0;
}}
.timeline-item::before {{
content: '';
position: absolute;
left: -7px;
top: 4px;
width: 12px;
height: 12px;
border-radius: 50%;
background: #7C3AED;
box-shadow: 0 0 12px rgba(124,58,237,0.5);
}}
.timeline-item.active::before {{
background: #22C55E;
box-shadow: 0 0 12px rgba(34,197,94,0.5);
animation: pulse-dot 1.5s infinite;
}}
@keyframes pulse-dot {{
0%, 100% {{ box-shadow: 0 0 12px rgba(34,197,94,0.5); }}
50% {{ box-shadow: 0 0 24px rgba(34,197,94,0.8); }}
}}
.timeline-title {{
font-weight: 600;
font-size: 0.9rem;
color: {text};
}}
.timeline-desc {{
font-size: 0.8rem;
color: {text_secondary};
margin-top: 2px;
}}
.weather-widget {{
background: linear-gradient(135deg, rgba(6,182,212,0.12), rgba(124,58,237,0.12));
border: 1px solid rgba(6,182,212,0.2);
border-radius: 20px;
padding: 28px;
animation: fade-in-up 0.5s ease forwards;
position: relative;
overflow: hidden;
}}
.weather-widget::after {{
content: '';
position: absolute;
top: -50%;
right: -30%;
width: 200px;
height: 200px;
border-radius: 50%;
background: rgba(6,182,212,0.06);
pointer-events: none;
}}
.weather-temp {{
font-size: 3.5rem;
font-weight: 800;
background: linear-gradient(135deg, #06B6D4, #7C3AED);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
line-height: 1;
}}
.weather-stat {{
background: rgba(255,255,255,0.05);
border-radius: 12px;
padding: 10px 14px;
text-align: center;
}}
.weather-stat-label {{
font-size: 0.7rem;
color: {text_secondary};
text-transform: uppercase;
letter-spacing: 0.5px;
}}
.weather-stat-value {{
font-size: 1rem;
font-weight: 600;
color: {text};
margin-top: 2px;
}}
.news-card {{
background: {card_bg};
border: 1px solid {card_border};
border-radius: 14px;
padding: 18px 20px;
margin-bottom: 12px;
transition: all 0.3s ease;
animation: fade-in-up 0.5s ease forwards;
}}
.news-card:hover {{
border-color: rgba(124,58,237,0.3);
transform: translateX(4px);
box-shadow: 0 8px 24px {shadow_color};
}}
.news-title {{
font-weight: 600;
font-size: 0.95rem;
color: {text};
margin-bottom: 6px;
}}
.news-snippet {{
font-size: 0.82rem;
color: {text_secondary};
line-height: 1.5;
margin-bottom: 10px;
}}
.news-link {{
display: inline-flex;
align-items: center;
gap: 4px;
font-size: 0.8rem;
color: #7C3AED;
text-decoration: none;
font-weight: 500;
transition: color 0.2s;
}}
.news-link:hover {{ color: #06B6D4; }}
.metric-card {{
background: {card_bg};
border: 1px solid {card_border};
border-radius: 14px;
padding: 16px;
text-align: center;
transition: all 0.3s ease;
}}
.metric-card:hover {{
border-color: rgba(124,58,237,0.3);
transform: translateY(-2px);
}}
.metric-value {{
font-size: 1.6rem;
font-weight: 800;
background: linear-gradient(135deg, #7C3AED, #06B6D4);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
background-clip: text;
}}
.metric-label {{
font-size: 0.72rem;
color: {text_secondary};
text-transform: uppercase;
letter-spacing: 0.5px;
margin-top: 4px;
}}
.sidebar-section {{
margin-bottom: 24px;
}}
.sidebar-label {{
font-size: 0.7rem;
text-transform: uppercase;
letter-spacing: 1px;
color: {text_secondary};
margin-bottom: 8px;
font-weight: 600;
}}
.sidebar-value {{
font-size: 0.88rem;
color: {text};
font-weight: 500;
}}
.sidebar-tool-item {{
display: flex;
align-items: center;
gap: 8px;
padding: 6px 0;
font-size: 0.85rem;
color: {text};
}}
.status-dot {{
display: inline-block;
width: 8px;
height: 8px;
border-radius: 50%;
background: #22C55E;
box-shadow: 0 0 8px rgba(34,197,94,0.6);
animation: pulse-dot 2s infinite;
}}
.status-dot.offline {{
background: #EF4444;
box-shadow: 0 0 8px rgba(239,68,68,0.6);
}}
.feature-card {{
background: {card_bg};
border: 1px solid {card_border};
border-radius: 16px;
padding: 20px;
text-align: center;
transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
cursor: default;
position: relative;
overflow: hidden;
}}
.feature-card::after {{
content: '';
position: absolute;
inset: 0;
border-radius: 16px;
padding: 1px;
background: linear-gradient(135deg, transparent 40%, rgba(124,58,237,0.3));
-webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
-webkit-mask-composite: xor;
mask-composite: exclude;
opacity: 0;
transition: opacity 0.35s ease;
}}
.feature-card:hover {{
transform: translateY(-5px);
box-shadow: 0 20px 40px {shadow_color};
}}
.feature-card:hover::after {{ opacity: 1; }}
.feature-icon {{
font-size: 2.2rem;
margin-bottom: 10px;
display: block;
}}
.feature-name {{
font-weight: 600;
font-size: 0.9rem;
color: {text};
}}
.feature-desc {{
font-size: 0.75rem;
color: {text_secondary};
margin-top: 4px;
}}
.action-btn {{
display: flex;
align-items: center;
gap: 8px;
width: 100%;
padding: 10px 14px;
border-radius: 10px;
border: 1px solid {card_border};
background: {card_bg};
color: {text};
font-size: 0.85rem;
cursor: pointer;
transition: all 0.25s ease;
text-align: left;
}}
.action-btn:hover {{
border-color: rgba(124,58,237,0.4);
background: rgba(124,58,237,0.08);
transform: translateX(3px);
}}
.copy-btn {{
background: rgba(255,255,255,0.06);
border: 1px solid rgba(255,255,255,0.1);
color: {text_secondary};
padding: 4px 12px;
border-radius: 6px;
font-size: 0.75rem;
cursor: pointer;
transition: all 0.2s ease;
}}
.copy-btn:hover {{
background: rgba(124,58,237,0.15);
color: #7C3AED;
border-color: rgba(124,58,237,0.3);
}}
.monitor-row {{
display: flex;
justify-content: space-between;
align-items: center;
padding: 8px 0;
border-bottom: 1px solid {card_border};
}}
.monitor-row:last-child {{ border-bottom: none; }}
.monitor-label {{
font-size: 0.8rem;
color: {text_secondary};
}}
.monitor-value {{
font-size: 0.82rem;
font-weight: 600;
color: {text};
}}
.footer-section {{
text-align: center;
padding: 32px 0 16px;
border-top: 1px solid {card_border};
margin-top: 32px;
}}
.footer-tech {{
display: flex;
justify-content: center;
flex-wrap: wrap;
gap: 16px;
margin: 12px 0;
}}
.footer-badge {{
background: {card_bg};
border: 1px solid {card_border};
border-radius: 20px;
padding: 6px 16px;
font-size: 0.78rem;
color: {text_secondary};
transition: all 0.2s ease;
}}
.footer-badge:hover {{
border-color: rgba(124,58,237,0.4);
color: {text};
}}
.footer-links {{
display: flex;
justify-content: center;
gap: 16px;
margin-top: 12px;
}}
.footer-link {{
color: {text_secondary};
text-decoration: none;
font-size: 1.2rem;
transition: color 0.2s, transform 0.2s;
display: inline-block;
}}
.footer-link:hover {{
color: #7C3AED;
transform: translateY(-2px);
}}
.streamlit-expanderHeader {{
font-size: 0.9rem !important;
font-weight: 600 !important;
}}
.stSpinner > div > div {{
border-top-color: #7C3AED !important;
}}
.stToast {{
background: {bg_secondary} !important;
border: 1px solid {card_border} !important;
border-radius: 12px !important;
color: {text} !important;
}}
.stProgress > div > div > div {{
background: linear-gradient(90deg, #7C3AED, #06B6D4) !important;
}}
@keyframes soft-pulse {{
0%, 100% {{ opacity: 0.6; }}
50% {{ opacity: 1; }}
}}
.stChatFloatingInputContainer {{ z-index: 100; }}
blockquote {{
border-left: 3px solid #7C3AED !important;
padding-left: 14px !important;
margin: 8px 0 !important;
}}
code {{
background: {code_bg} !important;
border-radius: 6px !important;
padding: 2px 6px !important;
font-size: 0.85em !important;
}}
pre {{
background: {code_bg} !important;
border-radius: 12px !important;
border: 1px solid {card_border} !important;
padding: 16px !important;
}}
</style>"""


# ═══════════════════════════════════════════════════════════════
# 5. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def inject_css(css_string: str):
    """Reliably inject CSS without markdown code-block issues."""
    try:
        st.html(css_string)
    except AttributeError:
        components.html(css_string, height=0)


def add_timeline(title: str, desc: str):
    st.session_state.timeline.append({
        "title": title,
        "desc": desc,
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "active": True
    })
    for i in range(len(st.session_state.timeline) - 1):
        st.session_state.timeline[i]["active"] = False


def get_detailed_weather(city: str):
    try:
        API_KEY = os.getenv("OPEN_WEATHER_API_KEY")
        if not API_KEY:
            return None
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            d = resp.json()
            return {
                "city": city.title(),
                "temp": d["main"]["temp"],
                "feels_like": d["main"]["feels_like"],
                "temp_min": d["main"]["temp_min"],
                "temp_max": d["main"]["temp_max"],
                "humidity": d["main"]["humidity"],
                "pressure": d["main"]["pressure"],
                "wind_speed": d["wind"]["speed"],
                "description": d["weather"][0]["description"].title(),
                "icon_code": d["weather"][0]["icon"],
            }
    except Exception:
        pass
    return None


def weather_icon_from_code(code: str) -> str:
    mapping = {
        "01": "☀️", "02": "⛅", "03": "☁️", "04": "☁️",
        "09": "🌧️", "10": "🌦️", "11": "⛈️", "13": "❄️",
        "50": "🌫️",
    }
    return mapping.get(code[:2], "🌤️")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def get_chat_export() -> str:
    lines = [
        "🌍 CityMind AI Agent — Conversation Export",
        f"📅 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 55, ""
    ]
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"👤 You:\n{msg.content}\n")
        elif isinstance(msg, AIMessage):
            lines.append(f"🤖 CityMind:\n{msg.content}\n")
        elif isinstance(msg, ToolMessage):
            lines.append(f"🔧 Tool Result:\n{msg.content}\n")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 6. SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "messages": [],
        "agent_state": "idle",
        "pending_tool_call": None,
        "timeline": [],
        "stop_requested": False,
        "theme": "dark",
        "show_search": False,
        "last_weather_city": None,
        "current_action": "Waiting for input...",
        "metrics": {
            "total_messages": 0,
            "tool_calls": 0,
            "weather_requests": 0,
            "news_searches": 0,
            "response_times": [],
            "start_time": time.time(),
        },
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ═══════════════════════════════════════════════════════════════
# 7. AGENT EXECUTION LOGIC
# ═══════════════════════════════════════════════════════════════

def do_thinking_step():
    if st.session_state.stop_requested:
        st.session_state.agent_state = "idle"
        st.session_state.stop_requested = False
        st.toast("⏹ Generation stopped", icon="⏹")
        return

    st.session_state.current_action = "🧠 AI is thinking..."
    add_timeline("AI Thinking", "Processing your query with Mistral Large...")

    try:
        llm_with_tools = llm.bind_tools(TOOL_LIST)
        t0 = time.time()
        response = llm_with_tools.invoke(st.session_state.messages)
        elapsed = time.time() - t0
        st.session_state.metrics["response_times"].append(elapsed)

        st.session_state.messages.append(response)

        if hasattr(response, "tool_calls") and response.tool_calls:
            tc = response.tool_calls[0]
            args = tc.get("args", {})
            if isinstance(args, str):
                args = json.loads(args)
            tc["args"] = args
            st.session_state.pending_tool_call = tc
            st.session_state.agent_state = "approval_pending"
            st.session_state.current_action = f"⏳ Awaiting approval for {tc['name']}"
            add_timeline("Tool Selected", f"{tc['name']}({args})")
        else:
            st.session_state.agent_state = "idle"
            st.session_state.current_action = "✅ Response ready"
            add_timeline("Final Answer", "AI generated the response")
    except Exception as e:
        st.session_state.messages.append(
            AIMessage(content=f"❌ **Error:** {str(e)}")
        )
        st.session_state.agent_state = "idle"
        st.session_state.current_action = "❌ Error occurred"
        st.toast("An error occurred", icon="❌")


def do_executing_step():
    tc = st.session_state.pending_tool_call
    tool_name = tc["name"]
    tool_args = tc.get("args", {})
    tool_id = tc["id"]

    st.session_state.current_action = f"⚡ Executing {tool_name}..."
    add_timeline("Tool Executed", f"Running {tool_name}...")

    try:
        tool_func = TOOLS_MAP[tool_name]
        result = tool_func.invoke(tool_args)
        tool_msg = ToolMessage(content=str(result), tool_call_id=tool_id)
        st.session_state.messages.append(tool_msg)

        st.session_state.metrics["tool_calls"] += 1
        if tool_name == "get_weather":
            st.session_state.metrics["weather_requests"] += 1
            st.session_state.last_weather_city = tool_args.get("city", "")
        elif tool_name == "get_news":
            st.session_state.metrics["news_searches"] += 1

        add_timeline("Result Received", "Tool returned data successfully")
    except Exception as e:
        tool_msg = ToolMessage(content=f"Error executing tool: {str(e)}", tool_call_id=tool_id)
        st.session_state.messages.append(tool_msg)
        add_timeline("Tool Error", str(e))

    st.session_state.pending_tool_call = None
    st.session_state.agent_state = "thinking"


# ═══════════════════════════════════════════════════════════════
# 8. UI RENDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div style="text-align:center; padding: 16px 0 8px;">
<div style="font-size:2.4rem; margin-bottom:4px;">🌍</div>
<div style="font-size:1.15rem; font-weight:700; color:#F8FAFC;">CityMind AI</div>
<div style="font-size:0.72rem; color:#94A3B8; margin-top:2px;">Intelligent City Assistant</div>
</div>
<hr style="border-color: rgba(255,255,255,0.08); margin: 16px 0;">
""", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-value">Mistral Large Latest</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-label" style="margin-top:14px;">Framework</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-value">LangChain</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-label" style="margin-top:14px;">Agent Type</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-value">Tool Calling Agent</div>', unsafe_allow_html=True)

        st.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 18px 0;">', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-label">Tools Enabled</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="sidebar-tool-item">✅ Weather Tool</div>
<div class="sidebar-tool-item">✅ Tavily News Search</div>
<div class="sidebar-tool-item">✅ Human Approval Middleware</div>
""", unsafe_allow_html=True)

        st.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 18px 0;">', unsafe_allow_html=True)

        state = st.session_state.agent_state
        status_map = {
            "idle": ("🟢 Online", False),
            "thinking": ("🟡 Thinking", False),
            "approval_pending": ("🟠 Awaiting Approval", False),
            "executing": ("🔵 Executing", False),
        }
        status_text, is_offline = status_map.get(state, ("🟢 Online", False))
        dot_class = "offline" if is_offline else ""
        st.markdown(f"""
<div class="sidebar-label">Status</div>
<div class="sidebar-value" style="display:flex;align-items:center;gap:8px;">
<span class="status-dot {dot_class}"></span>
{status_text}
</div>
""", unsafe_allow_html=True)

        st.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 18px 0;">', unsafe_allow_html=True)

        m = st.session_state.metrics
        avg_rt = (sum(m["response_times"]) / len(m["response_times"])) if m["response_times"] else 0
        dur = time.time() - m["start_time"]

        st.markdown('<div class="sidebar-label">Session Metrics</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="monitor-row">
<span class="monitor-label">Messages</span>
<span class="monitor-value">{m['total_messages']}</span>
</div>
<div class="monitor-row">
<span class="monitor-label">Tool Calls</span>
<span class="monitor-value">{m['tool_calls']}</span>
</div>
<div class="monitor-row">
<span class="monitor-label">Weather</span>
<span class="monitor-value">{m['weather_requests']}</span>
</div>
<div class="monitor-row">
<span class="monitor-label">News</span>
<span class="monitor-value">{m['news_searches']}</span>
</div>
<div class="monitor-row">
<span class="monitor-label">Avg Response</span>
<span class="monitor-value">{avg_rt:.1f}s</span>
</div>
<div class="monitor-row">
<span class="monitor-label">Duration</span>
<span class="monitor-value">{format_duration(dur)}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown('<hr style="border-color: rgba(255,255,255,0.08); margin: 18px 0;">', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-label">Actions</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True, key="btn_clear"):
                st.session_state.messages = []
                st.session_state.timeline = []
                st.session_state.agent_state = "idle"
                st.session_state.pending_tool_call = None
                st.session_state.last_weather_city = None
                st.session_state.current_action = "Waiting for input..."
                st.session_state.metrics = {
                    "total_messages": 0, "tool_calls": 0,
                    "weather_requests": 0, "news_searches": 0,
                    "response_times": [], "start_time": time.time(),
                }
                st.toast("Chat cleared", icon="🗑️")
                st.rerun()

        with col2:
            if st.session_state.agent_state != "idle":
                if st.button("⏹ Stop", use_container_width=True, key="btn_stop"):
                    st.session_state.stop_requested = True
                    st.rerun()

        theme_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
        if st.button(theme_label, use_container_width=True, key="btn_theme"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

        if st.session_state.messages:
            st.download_button(
                "📥 Export Chat",
                data=get_chat_export(),
                file_name=f"citymind_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="btn_export"
            )

        if st.button("🔍 Search Chat", use_container_width=True, key="btn_search_toggle"):
            st.session_state.show_search = not st.session_state.show_search


def render_hero():
    if st.session_state.messages:
        return

    st.markdown("""
<div style="position:relative; text-align:center; padding: 48px 20px 32px; overflow:hidden;">
<div class="particle" style="width:80px;height:80px;top:8%;left:15%;background:rgba(124,58,237,0.12);animation-delay:0s;"></div>
<div class="particle" style="width:60px;height:60px;top:55%;left:75%;background:rgba(6,182,212,0.1);animation-delay:2s;"></div>
<div class="particle" style="width:45px;height:45px;top:25%;left:82%;background:rgba(34,197,94,0.08);animation-delay:4s;"></div>
<div class="particle" style="width:55px;height:55px;top:65%;left:10%;background:rgba(245,158,11,0.08);animation-delay:1s;"></div>
<div class="particle" style="width:35px;height:35px;top:12%;left:60%;background:rgba(124,58,237,0.1);animation-delay:3s;"></div>
<div class="fade-in">
<div style="font-size:4rem; margin-bottom:12px;">🌍</div>
</div>
<h1 class="gradient-title fade-in fade-in-d1">CityMind AI Agent</h1>
<p class="fade-in fade-in-d2" style="font-size:1.1rem;color:#94A3B8;max-width:620px;margin:16px auto 0;line-height:1.7;">
Your intelligent AI assistant that can search live news, check weather,
use tools, and reason before answering.
</p>
</div>
""", unsafe_allow_html=True)

    features = [
        ("🌦️", "Weather Tool", "Real-time weather data"),
        ("📰", "News Search", "Live news via Tavily"),
        ("🤖", "AI Agent", "Mistral Large reasoning"),
        ("⚡", "Tool Calling", "Human-approved execution"),
    ]
    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(features):
        with cols[i]:
            st.markdown(f"""
<div class="feature-card fade-in fade-in-d{i+1}">
<span class="feature-icon">{icon}</span>
<div class="feature-name">{name}</div>
<div class="feature-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)


def render_tool_result_card(msg: ToolMessage):
    content = msg.content

    if content.startswith("Weather in"):
        city = content.split("Weather in ")[1].split(":")[0].strip()
        st.markdown(f"""
<div class="tool-call-card">
<span class="tool-icon">🌦️</span>
<span>Weather data received for <strong>{city}</strong></span>
</div>
""", unsafe_allow_html=True)

        detailed = get_detailed_weather(city)
        if detailed:
            icon = weather_icon_from_code(detailed["icon_code"])
            st.markdown(f"""
<div class="weather-widget">
<div style="display:flex; align-items:center; gap:20px; margin-bottom:20px; position:relative; z-index:1;">
<div style="font-size:3.5rem;">{icon}</div>
<div>
<div style="font-size:1rem; color:#94A3B8; margin-bottom:2px;">{detailed['city']}</div>
<div class="weather-temp">{detailed['temp']:.0f}°C</div>
<div style="font-size:0.9rem; color:#94A3B8; margin-top:4px;">{detailed['description']}</div>
</div>
</div>
<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px; position:relative; z-index:1;">
<div class="weather-stat">
<div class="weather-stat-label">Feels Like</div>
<div class="weather-stat-value">{detailed['feels_like']:.0f}°</div>
</div>
<div class="weather-stat">
<div class="weather-stat-label">Humidity</div>
<div class="weather-stat-value">{detailed['humidity']}%</div>
</div>
<div class="weather-stat">
<div class="weather-stat-label">Wind</div>
<div class="weather-stat-value">{detailed['wind_speed']} m/s</div>
</div>
<div class="weather-stat">
<div class="weather-stat-label">Pressure</div>
<div class="weather-stat-value">{detailed['pressure']} hPa</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)
        else:
            with st.expander("🔧 Raw Weather Data", expanded=False):
                st.code(content, language="text")
        return

    if content.startswith("Latest news in"):
        city = content.split("Latest news in ")[1].split(":")[0].strip()
        st.markdown(f"""
<div class="tool-call-card">
<span class="tool-icon">📰</span>
<span>News results received for <strong>{city}</strong></span>
</div>
""", unsafe_allow_html=True)

        items = content.split("\n\n")[1:]
        for item in items:
            lines = item.strip().split("\n")
            if len(lines) >= 3:
                title = lines[0].lstrip("-").strip()
                url_line = lines[1].strip()
                url = url_line.replace("🔗", "").strip()
                snippet = lines[2].replace("📰", "").strip().rstrip("...")
                st.markdown(f"""
<div class="news-card">
<div class="news-title">{title}</div>
<div class="news-snippet">{snippet}</div>
<a href="{url}" target="_blank" class="news-link">🔗 Open Article →</a>
</div>
""", unsafe_allow_html=True)
        return

    with st.expander("🔧 Tool Result", expanded=False):
        st.code(content, language="text")


def render_chat():
    search_term = ""
    if st.session_state.show_search:
        search_term = st.text_input(
            "🔍 Search in chat...",
            placeholder="Type to filter messages...",
            key="chat_search_input",
            label_visibility="collapsed"
        )

    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.messages:
            if search_term and search_term.lower() not in msg.content.lower():
                continue

            if isinstance(msg, HumanMessage):
                with st.chat_message("user", avatar="🌍"):
                    st.markdown(msg.content)

            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant", avatar="🤖"):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tname = tc["name"]
                            targs = tc.get("args", {})
                            if isinstance(targs, str):
                                targs = json.loads(targs)
                            icon = "🌦️" if tname == "get_weather" else "📰"
                            args_str = ", ".join(f"{k}={v}" for k, v in targs.items())
                            st.markdown(f"""
<div class="tool-call-card">
<span class="tool-icon">{icon}</span>
<span>Calling <strong>{tname}</strong> with <code>{args_str}</code></span>
</div>
""", unsafe_allow_html=True)
                    if msg.content:
                        st.markdown(msg.content)
                        st.markdown("""
<div style="margin-top:8px;">
<button class="copy-btn" onclick="
var text = this.closest('.stChatMessageContent').innerText;
navigator.clipboard.writeText(text);
this.textContent='✅ Copied!';
setTimeout(()=>this.textContent='📋 Copy', 2000);
">📋 Copy</button>
</div>
""", unsafe_allow_html=True)

            elif isinstance(msg, ToolMessage):
                with st.chat_message("assistant", avatar="🔧"):
                    render_tool_result_card(msg)

        state = st.session_state.agent_state
        if state == "thinking":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("""
<div class="typing-indicator">
<div class="typing-dots"><span></span><span></span><span></span></div>
<span>AI is thinking...</span>
</div>
""", unsafe_allow_html=True)
        elif state == "executing":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("""
<div class="typing-indicator">
<div class="typing-dots"><span></span><span></span><span></span></div>
<span>⚡ Executing tool...</span>
</div>
""", unsafe_allow_html=True)
        elif state == "approval_pending":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("""
<div class="typing-indicator">
<span style="animation: soft-pulse 1.5s infinite;">⏳</span>
<span>Waiting for your approval...</span>
</div>
""", unsafe_allow_html=True)


def render_approval_panel():
    if st.session_state.agent_state != "approval_pending":
        return
    tc = st.session_state.pending_tool_call
    if not tc:
        return

    tool_name = tc["name"]
    args = tc.get("args", {})
    if isinstance(args, str):
        args = json.loads(args)

    icon = "🌦️" if tool_name == "get_weather" else "📰"
    display_name = "Weather Tool" if tool_name == "get_weather" else "Tavily News Search"
    args_html = ""
    for k, v in args.items():
        args_html += f'<div class="monitor-row"><span class="monitor-label">{k}</span><span class="monitor-value" style="color:#06B6D4;">{v}</span></div>'

    st.markdown(f"""
<div class="approval-panel">
<div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
<span style="font-size:2rem;">{icon}</span>
<div>
<div style="font-size:1.1rem;font-weight:700;color:#F8FAFC;">Tool Approval Required</div>
<div style="font-size:0.82rem;color:#94A3B8;margin-top:2px;">
The agent wants to use a tool before responding
</div>
</div>
</div>
<div style="background:rgba(0,0,0,0.2);border-radius:12px;padding:16px;margin-bottom:24px;">
<div class="monitor-row" style="border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:12px;margin-bottom:8px;">
<span class="monitor-label">Tool Requested</span>
<span class="monitor-value" style="color:#7C3AED;">{display_name}</span>
</div>
<div style="font-size:0.75rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">Arguments</div>
{args_html}
</div>
</div>
""", unsafe_allow_html=True)

    col_a, col_d = st.columns(2)
    with col_a:
        if st.button("✅  Approve Tool Call", use_container_width=True, type="primary", key="btn_approve"):
            st.session_state.agent_state = "executing"
            st.toast("✅ Tool call approved", icon="✅")
            st.rerun()
    with col_d:
        if st.button("❌  Deny Tool Call", use_container_width=True, key="btn_deny"):
            tc = st.session_state.pending_tool_call
            deny_msg = ToolMessage(
                content="Tool call denied by user.",
                tool_call_id=tc["id"]
            )
            st.session_state.messages.append(deny_msg)
            st.session_state.pending_tool_call = None
            st.session_state.agent_state = "thinking"
            add_timeline("Tool Denied", "User rejected the tool call")
            st.toast("❌ Tool call denied", icon="❌")
            st.rerun()


def render_timeline():
    if not st.session_state.timeline:
        return

    with st.expander("📋 Execution Timeline", expanded=False):
        items_html = ""
        for item in st.session_state.timeline:
            active_class = " active" if item.get("active") else ""
            items_html += f"""
<div class="timeline-item{active_class}">
<div class="timeline-title">{item['title']}</div>
<div class="timeline-desc">{item['desc']} · {item['time']}</div>
</div>
"""
        st.markdown(f'<div class="timeline-container">{items_html}</div>', unsafe_allow_html=True)


def render_activity_monitor():
    state = st.session_state.agent_state
    state_display = {
        "idle": ("Idle", "#94A3B8"),
        "thinking": ("Processing", "#F59E0B"),
        "approval_pending": ("Awaiting Approval", "#F97316"),
        "executing": ("Executing Tool", "#06B6D4"),
    }
    state_text, state_color = state_display.get(state, ("Idle", "#94A3B8"))

    current_tool = "None"
    if st.session_state.pending_tool_call:
        current_tool = st.session_state.pending_tool_call["name"]

    m = st.session_state.metrics
    avg_rt = (sum(m["response_times"]) / len(m["response_times"])) if m["response_times"] else 0

    with st.expander("📊 Agent Activity Monitor", expanded=False):
        st.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border-radius:14px;padding:20px;border:1px solid rgba(255,255,255,0.06);">
<div class="monitor-row">
<span class="monitor-label">Agent Status</span>
<span class="monitor-value" style="color:{state_color};">● {state_text}</span>
</div>
<div class="monitor-row">
<span class="monitor-label">Selected Tool</span>
<span class="monitor-value">{current_tool}</span>
</div>
<div class="monitor-row">
<span class="monitor-label">Current Action</span>
<span class="monitor-value" style="font-size:0.78rem;">{st.session_state.get('current_action', 'Waiting...')}</span>
</div>
<div class="monitor-row">
<span class="monitor-label">Thinking State</span>
<span class="monitor-value">{state_text}</span>
</div>
<div class="monitor-row">
<span class="monitor-label">Last Response Time</span>
<span class="monitor-value">{avg_rt:.1f}s</span>
</div>
</div>
""", unsafe_allow_html=True)


def render_metrics_dashboard():
    m = st.session_state.metrics
    avg_rt = (sum(m["response_times"]) / len(m["response_times"])) if m["response_times"] else 0
    dur = time.time() - m["start_time"]

    metrics = [
        ("💬", m["total_messages"], "Total Messages"),
        ("⚡", m["tool_calls"], "Tool Calls"),
        ("🌦️", m["weather_requests"], "Weather Requests"),
        ("📰", m["news_searches"], "News Searches"),
        ("⏱️", f"{avg_rt:.1f}s", "Avg Response"),
        ("🕐", format_duration(dur), "Session Duration"),
    ]

    with st.expander("📈 Metrics Dashboard", expanded=False):
        cols = st.columns(3)
        for i, (icon, value, label) in enumerate(metrics):
            with cols[i % 3]:
                st.markdown(f"""
<div class="metric-card">
<div style="font-size:1.3rem;margin-bottom:6px;">{icon}</div>
<div class="metric-value">{value}</div>
<div class="metric-label">{label}</div>
</div>
""", unsafe_allow_html=True)


def render_footer():
    st.markdown("""
<div class="footer-section">
<div style="font-size:0.78rem;color:#94A3B8;">Powered by</div>
<div class="footer-tech">
<span class="footer-badge">🔗 LangChain</span>
<span class="footer-badge">🧠 Mistral AI</span>
<span class="footer-badge">🔍 Tavily Search</span>
<span class="footer-badge">🌤️ OpenWeather API</span>
<span class="footer-badge">⚡ Streamlit</span>
</div>
<div class="footer-links">
<a href="https://github.com" target="_blank" class="footer-link" title="GitHub">📂</a>
<a href="https://linkedin.com" target="_blank" class="footer-link" title="LinkedIn">💼</a>
<a href="mailto:hello@example.com" class="footer-link" title="Email">✉️</a>
</div>
<div style="font-size:0.7rem;color:#64748B;margin-top:12px;">
Built with ❤️ using Streamlit · CityMind AI Agent © 2025
</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 9. MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════

def main():
    init_session_state()

    # Inject CSS via st.html() — avoids markdown code-block bug
    css = get_css(st.session_state.theme)
    inject_css(css)

    # ── Execute Agent Steps ──
    if st.session_state.agent_state == "thinking":
        with st.spinner("🧠 AI is thinking..."):
            do_thinking_step()
        if st.session_state.agent_state != "approval_pending":
            st.rerun()

    elif st.session_state.agent_state == "executing":
        with st.spinner("⚡ Executing tool..."):
            do_executing_step()
        st.rerun()

    # ── Render Sidebar ──
    render_sidebar()

    # ── Main Content Area ──
    render_hero()
    render_chat()
    render_approval_panel()

    if st.session_state.timeline or st.session_state.metrics["total_messages"] > 0:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            render_timeline()
        with col_right:
            render_activity_monitor()

    render_metrics_dashboard()
    render_footer()

    # ── Chat Input ──
    if st.session_state.agent_state == "idle":
        user_input = st.chat_input(
            "Ask about any city — weather, news, or anything else...",
            key="main_chat_input"
        )
        if user_input and user_input.strip():
            st.session_state.messages.append(HumanMessage(content=user_input.strip()))
            st.session_state.metrics["total_messages"] += 1
            st.session_state.agent_state = "thinking"
            st.session_state.current_action = "🧠 AI is thinking..."
            add_timeline("User Query", user_input.strip()[:80] + ("..." if len(user_input.strip()) > 80 else ""))
            st.rerun()
    else:
        st.chat_input(
            "Agent is busy...",
            key="disabled_chat_input",
            disabled=True
        )


if __name__ == "__main__":
    main()