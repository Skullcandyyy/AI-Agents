"""
City Agent - Streamlit UI Implementation (CityMind 3.0 - Portfolio Grade)
This file is a complete, self-contained Streamlit application wrapping the logic from Agents.py.
"""

import os
import requests
import streamlit as st
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import time
import pydeck as pdk

# Load env variables
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from tavily import TavilyClient

# ═══════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & PREMIUM CSS
# ═══════════════════════════════════════════════════════════════
import random

# We expand the sidebar to show the new features while keeping the landing-page feel.
st.set_page_config(page_title="CityMind - AI Agent", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

if "bg_url" not in st.session_state:
    premium_bgs = [
        "https://images.unsplash.com/photo-1536098561742-ca998e48cbcc?auto=format&fit=crop&q=80&w=1600&h=900",
        "https://images.unsplash.com/photo-1499856871958-5b9627545d1a?auto=format&fit=crop&q=80&w=1600&h=900",
        "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?auto=format&fit=crop&q=80&w=1600&h=900",
        "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?auto=format&fit=crop&q=80&w=1600&h=900",
        "https://images.unsplash.com/photo-1506973035872-a4ec16b8e8d9?auto=format&fit=crop&q=80&w=1600&h=900",
        "https://images.unsplash.com/photo-1529655683826-aba9b3e77383?auto=format&fit=crop&q=80&w=1600&h=900"
    ]
    st.session_state.bg_url = random.choice(premium_bgs)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Base Reset & Fonts */
.stApp {
    background: #050B14;
    color: #F8FAFC;
    font-family: 'Outfit', sans-serif !important;
}

/* Hide default streamlit padding at the top */
.block-container {
    padding-top: 2rem !important;
    max-width: 1200px !important;
}

/* ───────────────────────────────────────────────────────── */
/* DYNAMIC BACKGROUND */
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(rgba(15,23,42,0.85), rgba(15,23,42,0.95)), url("DYNAMIC_BG_URL_HERE");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}
[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.15) !important;
    backdrop-filter: blur(10px) saturate(120%) !important;
    -webkit-backdrop-filter: blur(10px) saturate(120%) !important;
    border-right: 1px solid rgba(255,255,255,0.1) !important;
}
[data-testid="stSidebar"] button {
    height: 60px !important;
    width: 100% !important;
    white-space: nowrap !important;
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #cbd5e1 !important;
    border-radius: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stSidebar"] button:hover {
    border-color: #38BDF8 !important;
    background: rgba(56, 189, 248, 0.1) !important;
    color: #38BDF8 !important;
}
[data-testid="stSidebar"] button p {
    color: inherit !important;
    margin: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: transparent !important;
}
.stApp::before {
    display: none !important;
}

/* Chat Input Glassmorphism */
div[data-testid="stBottom"] {
    background: transparent !important;
}
div[data-testid="stBottom"] > div {
    background: transparent !important;
}
div[data-testid="stChatInput"] {
    background-color: rgba(15, 23, 42, 0.15) !important;
    backdrop-filter: blur(10px) saturate(120%) !important;
    -webkit-backdrop-filter: blur(10px) saturate(120%) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 30px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4) !important;
}
div[data-testid="stChatInput"] * {
    background-color: transparent !important;
}
div[data-testid="stChatInput"] textarea {
    color: white !important;
}

/* Header Glassmorphism */
[data-testid="stHeader"] {
    background-color: rgba(15, 23, 42, 0.15) !important;
    backdrop-filter: blur(10px) saturate(120%) !important;
    -webkit-backdrop-filter: blur(10px) saturate(120%) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
}
[data-testid="stHeader"] > div:first-child {
    background-color: transparent !important;
}

/* Custom AI Loading Animation */
[data-testid="stStatusWidget"] {
    visibility: hidden;
}
[data-testid="stStatusWidget"]::after {
    content: '⚡ AI Core is Processing...';
    visibility: visible;
    position: fixed;
    top: 15px;
    left: 360px;
    background: rgba(6, 182, 212, 0.15);
    border: 1px solid rgba(6, 182, 212, 0.4);
    color: #38BDF8;
    padding: 8px 18px;
    border-radius: 30px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.3);
    animation: ai-pulse 1.2s ease-in-out infinite alternate;
    z-index: 999999;
}
@keyframes ai-pulse {
    0% { box-shadow: 0 0 5px rgba(6, 182, 212, 0.2); transform: scale(1); }
    100% { box-shadow: 0 0 20px rgba(6, 182, 212, 0.8); transform: scale(1.02); }
}

/* ───────────────────────────────────────────────────────── */
/* HERO SECTION (PORTFOLIO GRADE) */
/* ───────────────────────────────────────────────────────── */
.hero-section {
    text-align: center;
    padding: 4rem 0 3rem 0;
    margin-bottom: 2rem;
    position: relative;
    z-index: 10;
}
.hero-badge {
    display: inline-block;
    padding: 8px 16px;
    background: rgba(6, 182, 212, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 30px;
    color: #38BDF8;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 20px;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
}
.hero-title {
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(to right, #F8FAFC, #94A3B8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    letter-spacing: -1px;
}
.hero-title span {
    background: linear-gradient(135deg, #7C3AED, #06B6D4, #EC4899);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 5s linear infinite;
}
.hero-subtitle {
    font-size: 1.2rem;
    color: #94A3B8;
    max-width: 600px;
    margin: 0 auto;
    font-weight: 300;
}

@keyframes shine {
    to { background-position: 200% center; }
}

/* ───────────────────────────────────────────────────────── */
/* GLASSMORPHIC DASHBOARD CARDS */
/* ───────────────────────────────────────────────────────── */
.glass-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
}
.glass-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 24px;
    padding: 25px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute; top: 0; left: -100%; width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
    transition: 0.5s;
}
.glass-card:hover::before {
    left: 100%;
}
.glass-card:hover {
    transform: translateY(-10px);
    border-color: rgba(124, 58, 237, 0.4);
    box-shadow: 0 20px 40px rgba(124, 58, 237, 0.2);
    background: rgba(255, 255, 255, 0.04);
}
.card-icon {
    font-size: 2rem;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #7C3AED, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 5px;
}
.card-label {
    font-size: 0.85rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* ───────────────────────────────────────────────────────── */
/* APP CONSOLE (CHAT AREA) */
/* ───────────────────────────────────────────────────────── */
/* Make the main chat area look like a floating terminal */
.stChatInputContainer {
    background: rgba(15, 23, 42, 0.15) !important;
    backdrop-filter: blur(10px) saturate(120%) !important;
    -webkit-backdrop-filter: blur(10px) saturate(120%) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: 30px !important;
    padding: 5px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5), 0 0 20px rgba(124,58,237,0.1) !important;
    transition: all 0.3s ease;
}
.stChatInputContainer:focus-within {
    border-color: rgba(6, 182, 212, 0.6) !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5), 0 0 30px rgba(6, 182, 212, 0.2) !important;
    transform: translateY(-2px);
}

.stChatMessage {
    background: transparent !important;
    padding: 1rem 0 !important;
}
.stChatMessage[data-testid="stChatMessageUser"] .stChatMessageContent {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(91,33,182,0.15));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 20px 20px 4px 20px;
    padding: 15px 25px;
    box-shadow: 0 4px 20px rgba(124,58,237,0.1);
    animation: slideInRight 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.stChatMessage[data-testid="stChatMessageAssistant"] .stChatMessageContent {
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px 20px 20px 4px;
    padding: 15px 25px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    animation: slideInLeft 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(30px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-30px); }
    to { opacity: 1; transform: translateX(0); }
}

/* ───────────────────────────────────────────────────────── */
/* HUMAN APPROVAL MODAL */
/* ───────────────────────────────────────────────────────── */
.approval-panel {
    background: rgba(11, 17, 33, 0.7);
    backdrop-filter: blur(40px);
    -webkit-backdrop-filter: blur(40px);
    border: 1px solid rgba(236, 72, 153, 0.4);
    border-radius: 24px;
    padding: 35px;
    margin: 30px 0;
    text-align: center;
    box-shadow: 0 20px 50px rgba(0,0,0,0.5), inset 0 0 0 1px rgba(255,255,255,0.1);
    animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
}
.approval-panel::before {
    content: '';
    position: absolute; top: 0; left: 0; width: 100%; height: 4px;
    background: linear-gradient(90deg, #7C3AED, #EC4899, #06B6D4);
    animation: bgPan 3s linear infinite;
    background-size: 200% auto;
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.95) translateY(20px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
}
@keyframes bgPan {
    to { background-position: 200% center; }
}
.approval-title {
    color: #F8FAFC;
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 10px;
    letter-spacing: -0.5px;
}

/* Buttons */
.stButton>button {
    border-radius: 16px;
    font-weight: 600;
    padding: 25px 20px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(255,255,255,0.1);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.9rem;
}
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #7C3AED, #06B6D4);
    border: none;
    color: white;
}
.stButton>button[kind="primary"]:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 15px 30px rgba(124,58,237,0.4);
}
.stButton>button:not([kind="primary"]):hover {
    transform: translateY(-5px) scale(1.02);
    border-color: #EC4899;
    box-shadow: 0 15px 30px rgba(236,72,153,0.2);
    background: rgba(236, 72, 153, 0.1);
    color: #EC4899;
}

/* ───────────────────────────────────────────────────────── */
/* NEWS CARDS */
/* ───────────────────────────────────────────────────────── */
.news-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-top: 20px;
}
.news-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 20px;
    transition: all 0.4s;
    text-decoration: none;
    color: inherit;
    display: flex;
    flex-direction: column;
}
.news-card:hover {
    transform: translateY(-8px);
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(6, 182, 212, 0.5);
    box-shadow: 0 15px 35px rgba(0,0,0,0.4), 0 0 20px rgba(6,182,212,0.1);
}
.news-title {
    font-weight: 700;
    font-size: 1.1rem;
    color: #F8FAFC;
    margin-bottom: 12px;
    line-height: 1.4;
}
.news-snippet {
    font-size: 0.9rem;
    color: #94A3B8;
    flex-grow: 1;
    line-height: 1.6;
}
</style>
"""
CSS = CSS.replace("DYNAMIC_BG_URL_HERE", st.session_state.bg_url)
st.markdown(CSS, unsafe_allow_html=True)

# Vanta.js 3D Particles Background (Transparent over Aurora)
st.markdown('''
<div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -2; pointer-events: none;" id="vanta-bg"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js"></script>
<script>
    setTimeout(function() {
        if (window.VANTA) {
            window.VANTA.NET({
                el: "#vanta-bg",
                mouseControls: true,
                touchControls: true,
                gyroControls: false,
                minHeight: 200.00,
                minWidth: 200.00,
                scale: 1.00,
                scaleMobile: 1.00,
                color: 0x7C3AED,
                backgroundColor: 0x000000,
                backgroundAlpha: 0.0,
                points: 10.00,
                maxDistance: 25.00,
                spacing: 20.00
            });
        }
    }, 1000);
</script>
''', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 2. STATE & UTILS
# ═══════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="you are a helpful city assistant")]
if "pending_tool_calls" not in st.session_state:
    st.session_state.pending_tool_calls = None
if "active_city" not in st.session_state:
    st.session_state.active_city = None
if "active_lat" not in st.session_state:
    st.session_state.active_lat = None
if "active_lon" not in st.session_state:
    st.session_state.active_lon = None
if "trigger_query" not in st.session_state:
    st.session_state.trigger_query = None

# Update active city based on user input heuristically
def detect_city(text):
    words = text.title().split()
    for word in words:
        if len(word) > 3:
            st.session_state.active_city = word
            break

# ═══════════════════════════════════════════════════════════════
# 3. TOOLS
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
    
    st.session_state.active_city = city
    st.session_state.active_lat = data["coord"]["lat"]
    st.session_state.active_lon = data["coord"]["lon"]

    return f"Weather in {city}: {temp}K, {desc}"

@tool
def get_weather_forecast(city: str) -> str:
    """Get a 7-day temperature forecast for a city"""
    st.session_state.active_city = city
    
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    geo_data = requests.get(geo_url).json()
    
    if not geo_data.get("results"):
        return f"Could not find coordinates for {city}"
        
    lat = geo_data["results"][0]["latitude"]
    lon = geo_data["results"][0]["longitude"]
    st.session_state.active_lat = lat
    st.session_state.active_lon = lon
    
    weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
    weather_data = requests.get(weather_url).json()
    
    daily = weather_data.get("daily", {})
    if not daily:
        return "Could not fetch forecast."
        
    return json.dumps({
        "type": "forecast_chart",
        "city": city,
        "dates": daily["time"],
        "max_temps": daily["temperature_2m_max"],
        "min_temps": daily["temperature_2m_min"]
    })

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@st.cache_data(ttl=3600, show_spinner=False)
def get_global_headlines():
    """Fetch global headlines once an hour to avoid rate limits"""
    try:
        response = tavily_client.search(
            query="Top global news headlines today",
            search_depth="fast",
            topic="news",
            days=1,
            max_results=3
        )
        return response.get("results", [])
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_india_headlines():
    """Fetch India headlines once an hour to avoid rate limits"""
    try:
        response = tavily_client.search(
            query="Top India news headlines today",
            search_depth="fast",
            topic="news",
            days=1,
            max_results=3
        )
        return response.get("results", [])
    except Exception:
        return []

@tool
def get_news(city: str) -> str:
    """"Get latest news of a city"""
    st.session_state.active_city = city
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="fast",
        topic="news",
        days=1,
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}."
    
    return json.dumps({
        "type": "news_cards",
        "city": city,
        "articles": results
    })

TOOLS_MAP = {
    "get_weather": get_weather,
    "get_weather_forecast": get_weather_forecast,
    "get_news": get_news
}

llm = ChatMistralAI(model="mistral-small-latest")
llm_with_tools = llm.bind_tools([get_weather, get_weather_forecast, get_news])


# ═══════════════════════════════════════════════════════════════
# 4. PORTFOLIO UI LAYOUT & SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('''
    <div style="background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(6,182,212,0.15)); border: 1px solid rgba(124,58,237,0.3); padding: 20px; border-radius: 15px; margin-bottom: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); backdrop-filter: blur(10px);">
        <h2 style="margin:0; font-size:1.6rem; font-weight:800; background: linear-gradient(135deg, #7C3AED, #06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Command Center</h2>
        <p style="margin:5px 0 0 0; font-size:0.85rem; color:#94A3B8;">System Status: <span style="color:#22C55E">Online</span></p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("### ⚙️ System Preferences")
    temp = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 🔌 API Connections")
    st.markdown('''
    <div style="display:flex; align-items:center; gap:10px; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
        <div style="width:10px; height:10px; border-radius:50%; background:#22C55E; box-shadow:0 0 10px #22C55E;"></div>
        <span style="font-size:0.9rem;">Mistral AI Core</span>
    </div>
    <div style="display:flex; align-items:center; gap:10px; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
        <div style="width:10px; height:10px; border-radius:50%; background:#22C55E; box-shadow:0 0 10px #22C55E;"></div>
        <span style="font-size:0.9rem;">OpenWeather Node</span>
    </div>
    <div style="display:flex; align-items:center; gap:10px; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
        <div style="width:10px; height:10px; border-radius:50%; background:#22C55E; box-shadow:0 0 10px #22C55E;"></div>
        <span style="font-size:0.9rem;">Tavily News Feed</span>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💾 Memory Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.messages = [SystemMessage(content="you are a helpful city assistant.")]
            st.session_state.pending_tool_calls = None
            st.session_state.active_city = None
            st.rerun()
    with col2:
        chat_data = json.dumps([{"role": m.type, "content": m.content} for m in st.session_state.messages if m.type in ("human", "ai")], indent=2)
        st.download_button(label="📥 Export", data=chat_data, file_name="chat_history.json", mime="application/json", use_container_width=True)


# --- HERO SECTION ---
st.markdown('''
<div class="hero-section">
    <div class="hero-title">Welcome to <span>CityMind</span></div>
    <div class="hero-subtitle">Experience the future of spatial intelligence. Powered by Mistral AI, OpenWeather, and real-time global news networks.</div>
</div>
''', unsafe_allow_html=True)

# --- DASHBOARD GRID ---
# Show the dashboard globally if a city is active, directly beneath the hero
if st.session_state.active_city:
    city_name = st.session_state.active_city.title()
    lat = f"{st.session_state.active_lat:.2f}" if st.session_state.active_lat else "--"
    lon = f"{st.session_state.active_lon:.2f}" if st.session_state.active_lon else "--"
    
    # Fetch AQI
    aqi_val, uv_val = "--", "--"
    aqi_color = "rgba(255,255,255,0.05)"
    if st.session_state.active_lat and st.session_state.active_lon:
        try:
            url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={st.session_state.active_lat}&longitude={st.session_state.active_lon}&current=european_aqi,uv_index"
            aqi_data = requests.get(url).json().get("current", {})
            aqi_val = aqi_data.get("european_aqi", "--")
            uv_val = aqi_data.get("uv_index", "--")
            
            if isinstance(aqi_val, (int, float)):
                if aqi_val < 50: aqi_color = "rgba(34,197,94,0.4)" # Green
                elif aqi_val < 100: aqi_color = "rgba(234,179,8,0.4)" # Yellow
                else: aqi_color = "rgba(239,68,68,0.4)" # Red
        except:
            pass
            
    st.markdown(f'''
    <div class="glass-grid">
        <div class="glass-card">
            <div class="card-icon">📍</div>
            <div class="card-value">{city_name}</div>
            <div class="card-label">Active Context</div>
        </div>
        <div class="glass-card">
            <div class="card-icon">🌐</div>
            <div class="card-value">{lat}°</div>
            <div class="card-label">Latitude</div>
        </div>
        <div class="glass-card">
            <div class="card-icon">🧭</div>
            <div class="card-value">{lon}°</div>
            <div class="card-label">Longitude</div>
        </div>
        <div class="glass-card" style="border-color: {aqi_color}; box-shadow: 0 10px 30px {aqi_color};">
            <div class="card-icon">🌬️</div>
            <div class="card-value">{aqi_val}</div>
            <div class="card-label">AQI (Air Quality)</div>
        </div>
        <div class="glass-card">
            <div class="card-icon">☀️</div>
            <div class="card-value">{uv_val}</div>
            <div class="card-label">UV Index</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Dynamic 4K Background
    if city_name:
        clean_city = city_name.replace(" ", "%20")
        st.markdown(f'''
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: linear-gradient(rgba(15,23,42,0.85), rgba(15,23,42,0.95)), url("https://loremflickr.com/1600/900/{clean_city},city/all");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(15, 23, 42, 0.15) !important;
            backdrop-filter: blur(10px) saturate(120%) !important;
            -webkit-backdrop-filter: blur(10px) saturate(120%) !important;
            border-right: 1px solid rgba(255,255,255,0.1) !important;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background-color: transparent !important;
        }}
        .stApp::before {{
            display: none !important;
        }}
        </style>
        ''', unsafe_allow_html=True)

# --- SUGGESTED CITIES ---
st.markdown("<br><h4 style='color:#cbd5e1; font-weight:400; text-align:center;'>🌍 Explore Global Destinations</h4>", unsafe_allow_html=True)

# Add some custom CSS just for these destination images
st.markdown("""
<style>
div[data-testid="stImage"] img {
    border-radius: 16px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    margin-bottom: 10px;
}
div[data-testid="stImage"] img:hover {
    transform: scale(1.05) translateY(-5px);
    box-shadow: 0 15px 30px rgba(6,182,212,0.3);
}
</style>
""", unsafe_allow_html=True)

suggestions = {
    "Tokyo": "https://images.unsplash.com/photo-1536098561742-ca998e48cbcc?auto=format&fit=crop&q=80&w=400&h=300",
    "Paris": "https://images.unsplash.com/photo-1499856871958-5b9627545d1a?auto=format&fit=crop&q=80&w=400&h=300",
    "New York": "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?auto=format&fit=crop&q=80&w=400&h=300",
    "Dubai": "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?auto=format&fit=crop&q=80&w=400&h=300",
    "Sydney": "https://images.unsplash.com/photo-1506973035872-a4ec16b8e8d9?auto=format&fit=crop&q=80&w=400&h=300",
    "London": "https://images.unsplash.com/photo-1529655683826-aba9b3e77383?auto=format&fit=crop&q=80&w=400&h=300"
}

s_cols = st.columns(6)
for i, (scity, img_url) in enumerate(suggestions.items()):
    with s_cols[i]:
        st.image(img_url, use_container_width=True)
        if st.button(scity, use_container_width=True, key=f"sug_{scity}"):
            st.session_state.trigger_query = f"What is the 7-day weather forecast and latest news in {scity}?"
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# --- GLOBAL HEADLINES ---
st.markdown("<h4 style='color:#cbd5e1; font-weight:400; text-align:center;'>🗞️ Daily Global Headlines</h4>", unsafe_allow_html=True)
global_news = get_global_headlines()
if global_news:
    html_cards = f'<div class="news-grid">'
    for art in global_news:
        title = art.get('title', 'News')
        url = art.get('url', '#')
        content = art.get('content', '')[:100] + '...'
        html_cards += f'''
        <a href="{url}" target="_blank" class="news-card">
            <div class="news-title">{title}</div>
            <div class="news-snippet">{content}</div>
        </a>
        '''
    html_cards += '</div>'
    st.html(html_cards)
else:
    st.markdown("<p style='text-align:center; color:#94A3B8;'>Could not fetch global headlines at this time.</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- INDIA HEADLINES ---
st.markdown("<h4 style='color:#cbd5e1; font-weight:400; text-align:center;'>🇮🇳 Daily India Headlines</h4>", unsafe_allow_html=True)
india_news = get_india_headlines()
if india_news:
    html_cards = f'<div class="news-grid">'
    for art in india_news:
        title = art.get('title', 'News')
        url = art.get('url', '#')
        content = art.get('content', '')[:100] + '...'
        html_cards += f'''
        <a href="{url}" target="_blank" class="news-card">
            <div class="news-title">{title}</div>
            <div class="news-snippet">{content}</div>
        </a>
        '''
    html_cards += '</div>'
    st.html(html_cards)
else:
    st.markdown("<p style='text-align:center; color:#94A3B8;'>Could not fetch India headlines at this time.</p>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 5. MAIN CHAT APP CONSOLE
# ═══════════════════════════════════════════════════════════════

def render_tool_result(content_str):
    try:
        data = json.loads(content_str)
        if data.get("type") == "forecast_chart":
            st.markdown(f"<h3 style='color:#06B6D4;'>📈 7-Day Forecast for {data['city']}</h3>", unsafe_allow_html=True)
            df = pd.DataFrame({
                "Date": data["dates"],
                "Max Temp (°C)": data["max_temps"],
                "Min Temp (°C)": data["min_temps"]
            }).set_index("Date")
            st.line_chart(df)
            return True
            
        elif data.get("type") == "news_cards":
            html_cards = f'<div class="news-grid">'
            for art in data["articles"]:
                title = art.get('title', 'News')
                url = art.get('url', '#')
                content = art.get('content', '')[:100] + '...'
                html_cards += f'''
                <a href="{url}" target="_blank" class="news-card">
                    <div class="news-title">{title}</div>
                    <div class="news-snippet">{content}</div>
                </a>
                '''
            html_cards += '</div>'
            st.html(html_cards)
            return True
    except:
        pass
    return False

# We wrap the chat in an empty container. Streamlit automatically applies our CSS to stChatMessage
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            continue
        
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        
        if isinstance(msg, ToolMessage):
            with st.chat_message("assistant", avatar="✨"):
                if not render_tool_result(msg.content):
                    st.info(f"**Data Retrieved:**\n\n{msg.content}")
            continue
            
        if isinstance(msg, AIMessage) and not msg.content and msg.tool_calls:
            continue
            
        with st.chat_message(role):
            st.markdown(msg.content)

# ── AUTOMATED TOOL EXECUTION & CHAT INPUT ──
user_input = st.chat_input("Initialize query (e.g. 'What is the forecast for Tokyo?')...")

if st.session_state.get("trigger_query"):
    user_input = st.session_state.trigger_query
    st.session_state.trigger_query = None

if user_input:
    detect_city(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.status("🧠 Processing neural pathways...", expanded=True) as status:
            time.sleep(0.3)
            response = llm_with_tools.invoke(st.session_state.messages)
            st.session_state.messages.append(response)
            
            while response.tool_calls:
                st.write(f"⚡ Automatically executing {len(response.tool_calls)} tool(s)...")
                for tcall in response.tool_calls:
                    st.write(f"   ↳ Retrieving data from `{tcall['name']}`...")
                    func = TOOLS_MAP.get(tcall["name"])
                    if func:
                        result = func.invoke(tcall["args"])
                        st.session_state.messages.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tcall["id"]
                        ))
                
                st.write("🔄 Synthesizing retrieved intelligence...")
                response = llm_with_tools.invoke(st.session_state.messages)
                st.session_state.messages.append(response)
            
            status.update(label="✅ Query Resolved", state="complete", expanded=False)
        
        # After completing the loop and getting the final AI message, rerun to render tool UI blocks
        st.rerun()