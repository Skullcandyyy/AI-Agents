"""
🌍 CityMind AI Agent — Premium Streamlit UI
Fixed: repeated approval prompts, news not showing, improved responsiveness.
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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from tavily import TavilyClient

st.set_page_config(
    page_title="CityMind AI Agent",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# 2. TOOL DEFINITIONS
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
        url   = r.get("url", "")
        snippet = r.get("content", "")[:150]
        # Use simple ASCII delimiters so parsing is reliable
        news_list.append(f"TITLE::{title}||URL::{url}||SNIPPET::{snippet}")
    return "NEWSFEED::" + city + "\n" + "\n".join(news_list)

TOOLS_MAP = {"get_weather": get_weather, "get_news": get_news}
TOOL_LIST = [get_weather, get_news]

# ═══════════════════════════════════════════════════════════════
# 3. LLM SETUP — Multi-provider with throttle & cache
# ═══════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────
# FREE MISTRAL MODELS (no credit card needed on La Plateforme)
# ───────────────────────────────────────────────────────────────
# Provider catalogue: key → (display_label, model_id, sdk_pkg)
PROVIDER_CATALOGUE = {
    # ══ FREE MISTRAL MODELS ══
    "mistral_nemo":   ("⭐ Mistral Nemo [FREE • BEST]",      "open-mistral-nemo",     "mistral"),
    "mistral_small":  ("🟢 Mistral Small [FREE]",             "mistral-small-latest",  "mistral"),
    "mixtral_8x7b":   ("🟢 Mixtral 8x7B [FREE]",             "open-mixtral-8x7b",     "mistral"),
    "mistral_7b":     ("🟢 Mistral 7B [FREE • fastest]",     "open-mistral-7b",       "mistral"),
    # ══ PAID MISTRAL MODELS ══
    "mistral_large":  ("🟣 Mistral Large [PAID]",           "mistral-large-latest",  "mistral"),
}

# Rate-limit throttle — seconds to wait between consecutive calls
MIN_CALL_GAP = {
    "mistral_nemo":  1.5,   # generous free limits
    "mistral_small": 2.0,
    "mixtral_8x7b":  2.0,
    "mistral_7b":    1.0,   # fastest free model
    "mistral_large": 5.0,   # tight free-tier quota
}

DEFAULT_PROVIDER = "mistral_nemo"   # ← best free model


def build_llm(provider_key: str):
    """Instantiate the LLM for the given provider key."""
    _, model_id, pkg = PROVIDER_CATALOGUE[provider_key]
    if pkg == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model_id)
    elif pkg == "groq":
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model=model_id, temperature=0)
        except ImportError:
            st.error("Install langchain-groq:  pip install langchain-groq")
            raise
    elif pkg == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_id, temperature=0)
        except ImportError:
            st.error("Install langchain-google-genai:  pip install langchain-google-genai")
            raise
    raise ValueError(f"Unknown provider: {provider_key}")


@st.cache_resource(show_spinner=False)
def get_cached_llm(provider_key: str):
    """Cache the LLM object per provider so we don't rebuild on every rerun."""
    base_llm = build_llm(provider_key)
    return base_llm.bind_tools(TOOL_LIST)


def get_active_llm():
    """Return the LLM bound to tools for the currently selected provider."""
    key = st.session_state.get("provider_key", DEFAULT_PROVIDER)
    return get_cached_llm(key)


# ── Simple in-memory response cache (keyed on last user message) ───────────────
_RESPONSE_CACHE: dict = {}
CACHE_MAX = 30   # keep at most 30 cached answers


def _cache_key(msgs) -> str:
    """Build a cache key from the last human message + provider."""
    provider = st.session_state.get("provider_key", DEFAULT_PROVIDER)
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            return f"{provider}::{str(m.content)[:500]}"
    return ""


# Backward-compat alias
llm = build_llm(DEFAULT_PROVIDER)
llm_with_tools = llm.bind_tools(TOOL_LIST)


# ═══════════════════════════════════════════════════════════════
# 4. CSS — Responsive, Animated, Premium Dark/Light
# ═══════════════════════════════════════════════════════════════

def get_css(theme: str = "dark") -> str:
    if theme == "dark":
        bg            = "#07101F"
        text          = "#EDF2FF"
        text_sec      = "#8899BB"
        card_bg       = "rgba(255,255,255,0.04)"
        card_border   = "rgba(255,255,255,0.07)"
        card_hover    = "rgba(255,255,255,0.08)"
        input_bg      = "rgba(255,255,255,0.05)"
        input_border  = "rgba(124,58,237,0.4)"
        sidebar_bg    = "rgba(7,16,31,0.97)"
        code_bg       = "rgba(0,0,0,0.4)"
        scroll_thumb  = "rgba(124,58,237,0.35)"
        user_bubble   = "linear-gradient(135deg,#7C3AED 0%,#5B21B6 100%)"
        ai_bubble     = "rgba(255,255,255,0.05)"
        shadow        = "rgba(0,0,0,0.55)"
        glow          = "rgba(124,58,237,0.14)"
    else:
        bg            = "#EEF2FF"
        text          = "#0D1830"
        text_sec      = "#5A6B8A"
        card_bg       = "rgba(255,255,255,0.88)"
        card_border   = "rgba(0,0,0,0.07)"
        card_hover    = "rgba(255,255,255,0.98)"
        input_bg      = "rgba(255,255,255,0.95)"
        input_border  = "rgba(124,58,237,0.35)"
        sidebar_bg    = "rgba(238,242,255,0.97)"
        code_bg       = "rgba(0,0,0,0.05)"
        scroll_thumb  = "rgba(124,58,237,0.25)"
        user_bubble   = "linear-gradient(135deg,#7C3AED 0%,#5B21B6 100%)"
        ai_bubble     = "rgba(0,0,0,0.03)"
        shadow        = "rgba(0,0,0,0.1)"
        glow          = "rgba(124,58,237,0.07)"

    aurora1 = "rgba(124,58,237,0.18)" if theme == "dark" else "rgba(124,58,237,0.08)"
    aurora2 = "rgba(6,182,212,0.14)"  if theme == "dark" else "rgba(6,182,212,0.06)"
    aurora3 = "rgba(34,197,94,0.10)"  if theme == "dark" else "rgba(34,197,94,0.05)"
    aurora4 = "rgba(236,72,153,0.10)" if theme == "dark" else "rgba(236,72,153,0.04)"
    mesh_op  = "0.03" if theme == "dark" else "0.04"

    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*,*::before,*::after{{box-sizing:border-box;}}
html,body,[class*="css"]{{font-family:'Inter',sans-serif!important;}}
.stApp{{background:{bg};color:{text};transition:background .5s,color .5s;overflow-x:hidden;}}
html,body,[class*="st-"]{{color:{text}!important;}}

/* ══ ANIMATED AURORA BACKGROUND ══
   IMPORTANT: pseudo-elements are z-index:-1 so Streamlit content renders on top.
   background-color must NOT be set here — it would hide all content. */
.stApp::before{{
    content:'';
    position:fixed;
    inset:0;
    z-index:-2;
    pointer-events:none;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, {aurora1}, transparent 65%),
        radial-gradient(ellipse 60% 80% at 80% 80%, {aurora2}, transparent 60%),
        radial-gradient(ellipse 70% 50% at 60% 20%, {aurora3}, transparent 65%),
        radial-gradient(ellipse 50% 70% at 10% 70%, {aurora4}, transparent 60%);
    animation: aurora-move 18s ease-in-out infinite alternate;
}}
@keyframes aurora-move {{
    0%   {{ filter: hue-rotate(0deg) brightness(1); }}
    33%  {{ filter: hue-rotate(25deg) brightness(1.05); }}
    66%  {{ filter: hue-rotate(-15deg) brightness(0.95); }}
    100% {{ filter: hue-rotate(15deg) brightness(1.02); }}
}}

/* ── Mesh grid overlay ── */
.stApp::after{{
    content:'';
    position:fixed;
    inset:0;
    z-index:-1;
    pointer-events:none;
    background-image:
        linear-gradient(rgba(124,58,237,{mesh_op}) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,58,237,{mesh_op}) 1px, transparent 1px);
    background-size: 60px 60px;
    mask-image: radial-gradient(ellipse 90% 90% at 50% 50%, black 20%, transparent 100%);
    animation: mesh-drift 30s linear infinite;
}}
@keyframes mesh-drift {{
    0%   {{ transform: translate(0,0); }}
    50%  {{ transform: translate(-30px,-20px); }}
    100% {{ transform: translate(0,0); }}
}}

/* ── hide streamlit chrome ── */
#MainMenu,footer,.stDeployButton{{visibility:hidden;display:none!important;}}
header[data-testid="stHeader"]{{background:transparent!important;backdrop-filter:none!important;}}

/* ── scrollbar ── */
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:{scroll_thumb};border-radius:10px;transition:background .3s;}}
::-webkit-scrollbar-thumb:hover{{background:rgba(124,58,237,.6);}}

/* ── sidebar ── */
section[data-testid="stSidebar"]{{
    background:{sidebar_bg}!important;
    backdrop-filter:blur(40px) saturate(200%) brightness(1.08)!important;
    border-right:1px solid {card_border}!important;
    box-shadow: 4px 0 32px rgba(0,0,0,.25);
    transition: box-shadow .4s ease;
}}
section[data-testid="stSidebar"] *{{color:{text}!important;}}

/* ── sidebar button hover glow ── */
section[data-testid="stSidebar"] .stButton>button{{
    transition: all .3s cubic-bezier(.23,1,.32,1) !important;
    position: relative;
    overflow: hidden;
}}
section[data-testid="stSidebar"] .stButton>button::after{{
    content:'';
    position:absolute;inset:0;
    background:linear-gradient(135deg,rgba(124,58,237,.25),rgba(6,182,212,.15));
    opacity:0;transition:opacity .3s;
    border-radius:inherit;
}}
section[data-testid="stSidebar"] .stButton>button:hover{{transform:translateY(-2px) !important;box-shadow:0 8px 24px rgba(124,58,237,.3) !important;}}
section[data-testid="stSidebar"] .stButton>button:hover::after{{opacity:1;}}
section[data-testid="stSidebar"] .stButton>button:active{{transform:scale(0.97) !important;}}

/* ── glass card ── */
.glass-card{{
    background:{card_bg};backdrop-filter:blur(24px) saturate(160%);
    border:1px solid {card_border};border-radius:20px;padding:24px;
    transition:all .4s cubic-bezier(.23,1,.32,1);
    position:relative;overflow:hidden;will-change:transform;
}}
.glass-card::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,#7C3AED,#06B6D4,#22C55E,#F59E0B);
    background-size:200% auto;opacity:0;
    transition:opacity .4s;animation:shimmer 2.5s linear infinite;
}}
.glass-card::after{{
    content:'';position:absolute;inset:0;border-radius:20px;
    background:radial-gradient(circle at var(--mx,50%) var(--my,50%),rgba(124,58,237,.12) 0%,transparent 60%);
    opacity:0;transition:opacity .35s;
}}
.glass-card:hover{{background:{card_hover};border-color:rgba(124,58,237,.45);
    transform:translateY(-5px) scale(1.005);
    box-shadow:0 24px 64px {shadow}, 0 0 0 1px rgba(124,58,237,.12), 0 0 40px rgba(124,58,237,.08);}}
.glass-card:hover::before{{opacity:1;}}
.glass-card:hover::after{{opacity:1;}}

/* ── gradient title ── */
.gradient-title{{
    font-size:clamp(1.8rem,5vw,3.4rem);font-weight:900;
    background:linear-gradient(270deg,#7C3AED,#06B6D4,#22C55E,#F59E0B,#EC4899,#7C3AED);
    background-size:400% 400%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;animation:gradient-shift 7s ease infinite;
    line-height:1.15;letter-spacing:-.02em;
}}
@keyframes gradient-shift{{0%{{background-position:0% 50%;}}50%{{background-position:100% 50%;}}100%{{background-position:0% 50%;}}}}

/* ── entrance animations ── */
@keyframes fade-in-up{{from{{opacity:0;transform:translateY(28px) scale(.98);}}to{{opacity:1;transform:translateY(0) scale(1);}}}}
@keyframes fade-in-left{{from{{opacity:0;transform:translateX(-20px);}}to{{opacity:1;transform:translateX(0);}}}}
@keyframes zoom-in{{from{{opacity:0;transform:scale(.92);}}to{{opacity:1;transform:scale(1);}}}}
.fade-in{{animation:fade-in-up .7s cubic-bezier(.16,1,.3,1) forwards;opacity:0;}}
.fade-in-left{{animation:fade-in-left .6s cubic-bezier(.16,1,.3,1) forwards;opacity:0;}}
.zoom-in{{animation:zoom-in .6s cubic-bezier(.16,1,.3,1) forwards;opacity:0;}}
.fade-in-d1{{animation-delay:.08s;}}.fade-in-d2{{animation-delay:.18s;}}
.fade-in-d3{{animation-delay:.28s;}}.fade-in-d4{{animation-delay:.38s;}}
.fade-in-d5{{animation-delay:.48s;}}

/* ── particles ── */
.particle{{position:absolute;border-radius:50%;pointer-events:none;animation:particle-float 10s ease-in-out infinite;filter:blur(1px);}}
@keyframes particle-float{{
    0%,100%{{transform:translateY(0) translateX(0) rotate(0deg) scale(1);}}
    25%{{transform:translateY(-30px) translateX(20px) rotate(90deg) scale(1.1);}}
    50%{{transform:translateY(-15px) translateX(-15px) rotate(180deg) scale(.9);}}
    75%{{transform:translateY(-40px) translateX(10px) rotate(270deg) scale(1.05);}}
}}

/* ── shimmer ── */
@keyframes shimmer{{0%{{background-position:-200% center;}}100%{{background-position:200% center;}}}}

/* ── chat messages ── */
.stChatMessage[data-testid="stChatMessageAssistant"] .stChatMessageContent{{
    background:{ai_bubble};border-radius:6px 20px 20px 20px;
    border:1px solid {card_border};padding:16px 22px;
    animation:fade-in-up .4s ease forwards;
}}
.stChatMessage[data-testid="stChatMessageUser"] .stChatMessageContent{{
    background:{user_bubble};border-radius:20px 6px 20px 20px;padding:14px 22px;
    animation:fade-in-up .4s ease forwards;box-shadow:0 8px 24px rgba(124,58,237,.25);
}}
.stChatMessage[data-testid="stChatMessageUser"] .stChatMessageContent *{{color:#FFF!important;}}
.stChatMessage{{padding:4px 0;}}

/* ── chat input ── */
.stChatInputContainer{{
    border:1px solid {input_border}!important;border-radius:20px!important;
    background:{input_bg}!important;backdrop-filter:blur(24px)!important;
    transition:all .35s cubic-bezier(.23,1,.32,1)!important;
    box-shadow:0 4px 24px {glow}!important;
}}
.stChatInputContainer:focus-within{{
    border-color:#7C3AED!important;
    box-shadow:0 0 0 3px rgba(124,58,237,.18),0 8px 32px {glow}!important;
    transform:translateY(-2px)!important;
}}
.stChatInputTextArea{{color:{text}!important;background:transparent!important;font-family:'Inter',sans-serif!important;}}
.stChatInputTextArea::placeholder{{color:{text_sec}!important;}}

/* ── typing dots ── */
.typing-indicator{{display:flex;align-items:center;gap:12px;color:{text_sec};font-size:.9rem;padding:6px 0;}}
.typing-dots{{display:flex;gap:5px;}}
.typing-dots span{{
    width:9px;height:9px;border-radius:50%;
    background:linear-gradient(135deg,#7C3AED,#06B6D4);
    animation:dot-bounce 1.4s infinite both;
}}
.typing-dots span:nth-child(1){{animation-delay:0s;}}
.typing-dots span:nth-child(2){{animation-delay:.22s;}}
.typing-dots span:nth-child(3){{animation-delay:.44s;}}
@keyframes dot-bounce{{0%,80%,100%{{transform:scale(.35) translateY(0);opacity:.4;}}40%{{transform:scale(1) translateY(-6px);opacity:1;}}}}

/* ── tool call card ── */
.tool-call-card{{
    background:linear-gradient(135deg,rgba(124,58,237,.08),rgba(6,182,212,.06));
    border:1px solid rgba(124,58,237,.22);border-radius:14px;
    padding:14px 20px;margin:10px 0;display:flex;align-items:center;gap:12px;
    font-size:.88rem;animation:fade-in-up .4s ease forwards;transition:all .3s ease;
}}
.tool-call-card:hover{{border-color:rgba(124,58,237,.4);transform:translateX(3px);}}
.tool-call-card .tool-icon{{font-size:1.4rem;}}

/* ── approval panel ── */
.approval-panel{{
    background:{card_bg};backdrop-filter:blur(30px);
    border:1px solid rgba(124,58,237,.28);border-radius:24px;padding:36px;
    animation:zoom-in .5s cubic-bezier(.16,1,.3,1) forwards;
    box-shadow:0 0 80px {glow};position:relative;overflow:hidden;
}}
.approval-panel::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,#7C3AED,#06B6D4,#22C55E);
    background-size:200% auto;animation:shimmer 2s linear infinite;
}}

/* ── auto-approve badge ── */
.auto-approve-badge{{
    display:inline-flex;align-items:center;gap:6px;
    background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.3);
    border-radius:20px;padding:5px 14px;font-size:.78rem;color:#22C55E;font-weight:600;
    animation:fade-in-up .4s ease forwards;
}}

/* ── timeline ── */
.timeline-container{{padding-left:8px;}}
.timeline-item{{
    position:relative;padding-left:36px;padding-bottom:22px;
    border-left:2px solid rgba(124,58,237,.18);animation:fade-in-left .5s ease forwards;
}}
.timeline-item:last-child{{border-left-color:transparent;padding-bottom:0;}}
.timeline-item::before{{
    content:'';position:absolute;left:-7px;top:4px;width:13px;height:13px;
    border-radius:50%;background:#7C3AED;box-shadow:0 0 14px rgba(124,58,237,.55);
}}
.timeline-item.active::before{{background:#22C55E;box-shadow:0 0 16px rgba(34,197,94,.6);animation:pulse-dot 1.8s infinite;}}
@keyframes pulse-dot{{0%,100%{{box-shadow:0 0 12px rgba(34,197,94,.5);transform:scale(1);}}50%{{box-shadow:0 0 28px rgba(34,197,94,.9);transform:scale(1.2);}}}}
.timeline-title{{font-weight:700;font-size:.88rem;color:{text};}}
.timeline-desc{{font-size:.78rem;color:{text_sec};margin-top:3px;line-height:1.4;}}

/* ── weather widget ── */
.weather-widget{{
    background:linear-gradient(135deg,rgba(6,182,212,.1),rgba(124,58,237,.1),rgba(34,197,94,.05));
    border:1px solid rgba(6,182,212,.22);border-radius:24px;padding:32px;
    animation:fade-in-up .6s ease forwards;position:relative;overflow:hidden;transition:all .35s ease;
}}
.weather-widget::after{{
    content:'';position:absolute;top:-60%;right:-25%;width:220px;height:220px;
    border-radius:50%;background:radial-gradient(circle,rgba(6,182,212,.08) 0%,transparent 70%);pointer-events:none;
}}
.weather-widget:hover{{transform:translateY(-3px);box-shadow:0 20px 60px {shadow};}}
.weather-temp{{
    font-size:3.8rem;font-weight:900;
    background:linear-gradient(135deg,#06B6D4,#7C3AED);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    line-height:1;letter-spacing:-.02em;
}}
.weather-stat{{
    background:rgba(255,255,255,.06);backdrop-filter:blur(10px);
    border:1px solid rgba(255,255,255,.07);border-radius:14px;
    padding:12px 16px;text-align:center;transition:all .3s ease;
}}
.weather-stat:hover{{background:rgba(255,255,255,.1);transform:translateY(-2px);}}
.weather-stat-label{{font-size:.68rem;color:{text_sec};text-transform:uppercase;letter-spacing:.8px;font-weight:600;}}
.weather-stat-value{{font-size:1.05rem;font-weight:700;color:{text};margin-top:4px;}}

/* ── news cards ── */
.news-card{{
    background:{card_bg};border:1px solid {card_border};border-radius:16px;
    padding:20px 22px;margin-bottom:14px;
    transition:all .38s cubic-bezier(.23,1,.32,1);animation:fade-in-up .5s ease forwards;
    position:relative;overflow:hidden;
}}
.news-card::before{{
    content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
    background:linear-gradient(180deg,#7C3AED,#06B6D4,#22C55E);
    opacity:0;transition:opacity .35s;
}}
.news-card::after{{
    content:'';position:absolute;inset:0;
    background:radial-gradient(circle at 0% 50%,rgba(124,58,237,.07) 0%,transparent 60%);
    opacity:0;transition:opacity .35s;
}}
.news-card:hover{{
    border-color:rgba(124,58,237,.4);transform:translateX(8px);
    box-shadow:0 12px 36px {shadow}, -4px 0 20px rgba(124,58,237,.1);
}}
.news-card:hover::before{{opacity:1;}}
.news-card:hover::after{{opacity:1;}}
.news-title{{
    font-weight:700;font-size:.95rem;color:{text};margin-bottom:8px;
    line-height:1.4;transition:color .3s;
}}
.news-card:hover .news-title{{color:#7C3AED;}}
.news-snippet{{font-size:.82rem;color:{text_sec};line-height:1.6;margin-bottom:12px;}}
.news-link{{
    display:inline-flex;align-items:center;gap:5px;font-size:.8rem;
    color:#7C3AED;text-decoration:none;font-weight:600;
    padding:5px 12px;border-radius:8px;background:rgba(124,58,237,.08);
    transition:all .28s cubic-bezier(.23,1,.32,1);
    position:relative;overflow:hidden;
}}
.news-link::before{{
    content:'';position:absolute;inset:0;
    background:linear-gradient(135deg,rgba(124,58,237,.2),rgba(6,182,212,.2));
    opacity:0;transition:opacity .25s;border-radius:inherit;
}}
.news-link:hover{{color:#fff;transform:translateX(4px) scale(1.04);box-shadow:0 4px 16px rgba(124,58,237,.3);}}
.news-link:hover::before{{opacity:1;}}

/* ── metric cards ── */
.metric-card{{
    background:{card_bg};border:1px solid {card_border};border-radius:18px;
    padding:20px 16px;text-align:center;
    transition:all .38s cubic-bezier(.23,1,.32,1);animation:fade-in-up .5s ease forwards;
    position:relative;overflow:hidden;
}}
.metric-card::before{{
    content:'';position:absolute;top:-40%;left:-40%;
    width:180%;height:180%;
    background:conic-gradient(from 0deg at 50% 50%,transparent 0deg,rgba(124,58,237,.08) 60deg,transparent 120deg);
    animation:spin-glow 8s linear infinite;opacity:0;transition:opacity .4s;
}}
@keyframes spin-glow{{to{{transform:rotate(360deg);}}}}
.metric-card:hover{{border-color:rgba(124,58,237,.4);transform:translateY(-6px) scale(1.03);
    box-shadow:0 24px 52px {shadow}, 0 0 40px rgba(124,58,237,.1);background:{card_hover};}}
.metric-card:hover::before{{opacity:1;}}
.metric-value{{
    font-size:1.8rem;font-weight:900;
    background:linear-gradient(135deg,#7C3AED,#06B6D4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    letter-spacing:-.02em;
}}
.metric-label{{font-size:.7rem;color:{text_sec};text-transform:uppercase;letter-spacing:.8px;margin-top:5px;font-weight:600;}}
.metric-icon{{font-size:1.5rem;margin-bottom:8px;display:block;transition:transform .4s;}}
.metric-card:hover .metric-icon{{transform:scale(1.25) rotate(10deg);}}

/* ── sidebar labels ── */
.sidebar-label{{font-size:.68rem;text-transform:uppercase;letter-spacing:1.2px;color:{text_sec};margin-bottom:8px;font-weight:700;}}
.sidebar-value{{font-size:.88rem;color:{text};font-weight:600;}}
.sidebar-tool-item{{display:flex;align-items:center;gap:10px;padding:7px 0;font-size:.85rem;color:{text};transition:color .2s;}}
.sidebar-tool-item:hover{{color:#7C3AED;}}
.status-dot{{
    display:inline-block;width:9px;height:9px;border-radius:50%;
    background:#22C55E;box-shadow:0 0 10px rgba(34,197,94,.7);animation:pulse-dot 2.2s infinite;
}}
.status-dot.offline{{background:#EF4444;box-shadow:0 0 10px rgba(239,68,68,.7);}}
.status-dot.busy{{background:#F59E0B;box-shadow:0 0 10px rgba(245,158,11,.7);}}

/* ── feature cards ── */
.feature-card{{
    background:{card_bg};border:1px solid {card_border};border-radius:20px;padding:24px 18px;
    text-align:center;transition:all .4s cubic-bezier(.23,1,.32,1);
    position:relative;overflow:hidden;cursor:default;
    transform-style:preserve-3d;
}}
.feature-card::before{{
    content:'';position:absolute;inset:0;border-radius:20px;
    background:radial-gradient(circle at var(--mx,50%) var(--my,50%),rgba(124,58,237,.18) 0%,transparent 65%);
    opacity:0;transition:opacity .35s ease;
}}
.feature-card::after{{
    content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,#7C3AED,#06B6D4,#22C55E);
    background-size:200% auto;
    opacity:0;transition:opacity .4s;animation:shimmer 2s linear infinite;
}}
.feature-card:hover{{
    transform:translateY(-10px) rotateX(3deg) rotateY(-1deg);
    box-shadow:0 32px 64px {shadow}, 0 0 0 1px rgba(124,58,237,.2),
               0 0 60px rgba(124,58,237,.1);
    border-color:rgba(124,58,237,.35);
}}
.feature-card:hover::before{{opacity:1;}}
.feature-card:hover::after{{opacity:1;}}
.feature-icon{{
    font-size:2.4rem;margin-bottom:12px;display:block;
    transition:transform .4s cubic-bezier(.23,1,.32,1),filter .4s;
}}
.feature-card:hover .feature-icon{{
    transform:scale(1.3) rotate(-8deg) translateY(-4px);
    filter:drop-shadow(0 8px 16px rgba(124,58,237,.5));
}}
.feature-name{{font-weight:700;font-size:.9rem;color:{text};transition:color .3s;}}
.feature-card:hover .feature-name{{color:#7C3AED;}}
.feature-desc{{font-size:.75rem;color:{text_sec};margin-top:5px;line-height:1.5;}}

/* ── monitor rows ── */
.monitor-row{{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid {card_border};}}
.monitor-row:last-child{{border-bottom:none;}}
.monitor-label{{font-size:.8rem;color:{text_sec};}}
.monitor-value{{font-size:.82rem;font-weight:700;color:{text};}}

/* ── copy btn ── */
.copy-btn{{
    background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);
    color:{text_sec};padding:5px 14px;border-radius:8px;font-size:.75rem;
    cursor:pointer;transition:all .25s;font-family:'Inter',sans-serif;
}}
.copy-btn:hover{{background:rgba(124,58,237,.18);color:#7C3AED;border-color:rgba(124,58,237,.35);transform:scale(1.05);}}

/* ── footer ── */
.footer-section{{text-align:center;padding:36px 0 20px;border-top:1px solid {card_border};margin-top:36px;}}
.footer-tech{{display:flex;justify-content:center;flex-wrap:wrap;gap:10px;margin:14px 0;}}
.footer-badge{{
    background:{card_bg};border:1px solid {card_border};border-radius:24px;
    padding:6px 18px;font-size:.78rem;color:{text_sec};transition:all .3s;cursor:default;
}}
.footer-badge:hover{{border-color:rgba(124,58,237,.4);color:{text};transform:translateY(-2px);background:{card_hover};}}

/* ── streamlit button ripple & glow ── */
.stButton>button{{
    position:relative;overflow:hidden;
    transition:all .32s cubic-bezier(.23,1,.32,1) !important;
    font-family:'Inter',sans-serif !important;
}}
.stButton>button::after{{
    content:'';position:absolute;
    width:200px;height:200px;
    top:50%;left:50%;
    transform:translate(-50%,-50%) scale(0);
    background:rgba(255,255,255,.18);
    border-radius:50%;
    opacity:0;
    transition:transform .5s ease, opacity .5s ease;
}}
.stButton>button:active::after{{
    transform:translate(-50%,-50%) scale(2.5);
    opacity:0;transition:0s;
}}
.stButton[data-testid] > button[kind="primary"]{{
    background:linear-gradient(135deg,#7C3AED,#5B21B6) !important;
    box-shadow:0 4px 16px rgba(124,58,237,.35) !important;
}}
.stButton[data-testid] > button[kind="primary"]:hover{{
    transform:translateY(-2px) !important;
    box-shadow:0 10px 32px rgba(124,58,237,.45) !important;
}}

/* ── expander hover ── */
.streamlit-expanderHeader{{
    font-size:.9rem!important;font-weight:700!important;
    transition:color .25s,background .25s !important;
    border-radius:10px !important;
}}
.streamlit-expanderHeader:hover{{background:rgba(124,58,237,.06) !important;color:#7C3AED !important;}}

/* ── chat message hover glow ── */
.stChatMessage{{transition:transform .3s ease !important;}}
.stChatMessage:hover{{transform:scale(1.005) !important;}}

/* ── weather widget extra hover ── */
.weather-stat:hover{{
    background:rgba(124,58,237,.12) !important;
    border-color:rgba(124,58,237,.25) !important;
    box-shadow:0 8px 24px rgba(124,58,237,.15) !important;
    transform:translateY(-3px) !important;
}}

/* ── footer badge hover ── */
.footer-badge:hover{{
    border-color:rgba(124,58,237,.5) !important;
    box-shadow:0 4px 20px rgba(124,58,237,.2) !important;
}}

/* ── spinner ── */
/* ── spinner ── */
.stSpinner>div>div{{border-top-color:#7C3AED!important;}}
.stProgress>div>div>div{{background:linear-gradient(90deg,#7C3AED,#06B6D4)!important;}}
blockquote{{border-left:3px solid #7C3AED!important;padding-left:16px!important;margin:10px 0!important;}}
code{{background:{code_bg}!important;border-radius:6px!important;padding:2px 7px!important;font-size:.85em!important;}}
pre{{background:{code_bg}!important;border-radius:14px!important;border:1px solid {card_border}!important;padding:18px!important;}}
.stChatFloatingInputContainer{{z-index:100;}}
.stCheckbox label{{font-size:.85rem!important;font-weight:600!important;}}

/* ── responsive ── */
@media(max-width:768px){{
    .gradient-title{{font-size:1.8rem!important;}}
    .weather-temp{{font-size:2.8rem!important;}}
    .approval-panel{{padding:20px!important;}}
    .weather-widget{{padding:20px!important;}}
    .feature-card:hover{{transform:translateY(-5px) !important;}}
}}
@keyframes soft-pulse{{0%,100%{{opacity:.6;}}50%{{opacity:1;}}}}
</style>"""


# ═══════════════════════════════════════════════════════════════
# 5. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def inject_css(css_string: str):
    try:
        st.html(css_string)
    except AttributeError:
        components.html(css_string, height=0)


def inject_bg_animation():
    """Inject JS particle canvas + 3D tilt effects via st.html."""
    anim_html = """
<style>
#cm-canvas {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: -3; pointer-events: none; opacity: 0.5;
}
</style>
<canvas id="cm-canvas"></canvas>
<script>
(function(){
  const canvas = document.getElementById('cm-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
  resize();
  window.addEventListener('resize', resize);

  const COLORS = ['#7C3AED','#06B6D4','#22C55E','#F59E0B','#EC4899'];
  const particles = [];
  class Particle {
    constructor() { this.reset(true); }
    reset(init) {
      this.x = Math.random() * canvas.width;
      this.y = init ? Math.random() * canvas.height : canvas.height + 10;
      this.r = Math.random() * 2 + 0.6;
      this.vx = (Math.random() - 0.5) * 0.4;
      this.vy = -(Math.random() * 0.6 + 0.2);
      this.col = COLORS[Math.floor(Math.random() * COLORS.length)];
      this.alpha = Math.random() * 0.5 + 0.15;
      this.life = 0;
      this.maxL = Math.random() * 280 + 180;
    }
    update() {
      this.x += this.vx; this.y += this.vy; this.life++;
      if (this.life > this.maxL || this.y < -10) this.reset(false);
    }
    draw() {
      const t = this.life / this.maxL;
      const a = this.alpha * (t < 0.15 ? t/0.15 : t > 0.85 ? (1-t)/0.15 : 1);
      ctx.beginPath(); ctx.arc(this.x, this.y, this.r, 0, Math.PI*2);
      ctx.fillStyle = this.col; ctx.globalAlpha = a; ctx.fill(); ctx.globalAlpha = 1;
    }
  }
  for (let i = 0; i < 60; i++) particles.push(new Particle());

  function drawConnections() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i+1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d  = Math.sqrt(dx*dx + dy*dy);
        if (d < 100) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = particles[i].col;
          ctx.globalAlpha = (1 - d/100) * 0.1;
          ctx.lineWidth = 0.7; ctx.stroke(); ctx.globalAlpha = 1;
        }
      }
    }
  }

  let mx = -999, my = -999;
  document.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; });

  function drawSpotlight() {
    const g = ctx.createRadialGradient(mx, my, 0, mx, my, 200);
    g.addColorStop(0, 'rgba(124,58,237,0.06)');
    g.addColorStop(0.6, 'rgba(6,182,212,0.02)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(mx, my, 200, 0, Math.PI*2); ctx.fill();
  }

  function attractToMouse() {
    particles.forEach(p => {
      const dx = mx - p.x, dy = my - p.y;
      const d  = Math.sqrt(dx*dx + dy*dy);
      if (d < 140 && d > 1) { p.vx += (dx/d)*0.01; p.vy += (dy/d)*0.01; }
    });
  }

  // ── SHOOTING STARS ─────────────────────────────────────────
  const STAR_COLORS = [
    ['#7C3AED','#C4B5FD'],  // purple
    ['#06B6D4','#A5F3FC'],  // cyan
    ['#EC4899','#FBCFE8'],  // pink
    ['#22C55E','#A7F3D0'],  // green
    ['#F59E0B','#FDE68A'],  // amber
  ];

  class ShootingStar {
    constructor() { this.reset(); }
    reset() {
      // Start from top-right area, move toward bottom-left
      const side = Math.random();
      if (side < 0.6) {
        // from top edge
        this.x  = Math.random() * canvas.width * 1.2;
        this.y  = -10;
      } else {
        // from right edge
        this.x  = canvas.width + 10;
        this.y  = Math.random() * canvas.height * 0.5;
      }
      const angle  = (210 + Math.random() * 30) * Math.PI / 180; // ~210-240 deg (down-left)
      const speed  = Math.random() * 9 + 7;       // 7–16 px/frame
      this.vx      = Math.cos(angle) * speed;
      this.vy      = Math.sin(angle) * speed;
      this.len     = Math.random() * 140 + 80;    // tail length 80–220 px
      this.width   = Math.random() * 1.8 + 0.6;  // stroke width
      const pair   = STAR_COLORS[Math.floor(Math.random() * STAR_COLORS.length)];
      this.colHead = pair[1];
      this.colTail = pair[0];
      this.alpha   = 0;
      this.maxA    = Math.random() * 0.75 + 0.35;
      this.phase   = 'in';   // 'in' | 'travel' | 'out'
      this.travelFrames = Math.floor(Math.random() * 18 + 12);
      this.frame   = 0;
      this.active  = true;
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      this.frame++;
      if (this.phase === 'in') {
        this.alpha = Math.min(this.alpha + 0.12, this.maxA);
        if (this.alpha >= this.maxA) this.phase = 'travel';
      } else if (this.phase === 'travel') {
        if (this.frame > this.travelFrames) this.phase = 'out';
      } else {
        this.alpha -= 0.07;
        if (this.alpha <= 0) this.active = false;
      }
    }
    draw() {
      if (this.alpha <= 0) return;
      // Tail: gradient line from head (bright) to tail (transparent)
      const tx = this.x - this.vx / Math.hypot(this.vx, this.vy) * this.len;
      const ty = this.y - this.vy / Math.hypot(this.vx, this.vy) * this.len;
      const grad = ctx.createLinearGradient(this.x, this.y, tx, ty);
      grad.addColorStop(0, this.colHead);          // bright head
      grad.addColorStop(0.3, this.colTail + 'AA'); // mid-tail
      grad.addColorStop(1, 'transparent');          // fade out
      ctx.save();
      ctx.globalAlpha = this.alpha;
      ctx.strokeStyle = grad;
      ctx.lineWidth   = this.width;
      ctx.lineCap     = 'round';
      ctx.shadowColor = this.colHead;
      ctx.shadowBlur  = 8;
      ctx.beginPath();
      ctx.moveTo(this.x, this.y);
      ctx.lineTo(tx, ty);
      ctx.stroke();
      // Bright head dot
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.width + 0.5, 0, Math.PI * 2);
      ctx.fillStyle = this.colHead;
      ctx.shadowBlur = 14;
      ctx.fill();
      ctx.restore();
    }
  }

  const shootingStars = [];
  let nextStarIn = 80;  // frames until next star spawns
  function spawnStar() {
    shootingStars.push(new ShootingStar());
    // next star in 1.8–4s at 60fps ≈ 108–240 frames
    nextStarIn = Math.floor(Math.random() * 130 + 108);
  }

  function updateShootingStars() {
    nextStarIn--;
    if (nextStarIn <= 0) {
      spawnStar();
      // occasionally spawn a double-star burst
      if (Math.random() < 0.25) {
        setTimeout(() => shootingStars.push(new ShootingStar()), 220);
      }
    }
    for (let i = shootingStars.length - 1; i >= 0; i--) {
      shootingStars[i].update();
      shootingStars[i].draw();
      if (!shootingStars[i].active) shootingStars.splice(i, 1);
    }
  }

  function loop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawSpotlight();
    updateShootingStars();   // ← shooting stars (behind particles)
    attractToMouse(); drawConnections();
    particles.forEach(p => { p.update(); p.draw(); });
    requestAnimationFrame(loop);
  }
  loop();

  function setupTilt() {
    document.querySelectorAll('.feature-card, .metric-card').forEach(card => {
      if (card._tilt) return;
      card._tilt = true;
      card.addEventListener('mousemove', e => {
        const r  = card.getBoundingClientRect();
        const dx = (e.clientX - r.left - r.width/2)  / (r.width/2);
        const dy = (e.clientY - r.top  - r.height/2) / (r.height/2);
        card.style.transform = `perspective(600px) rotateX(${-dy*7}deg) rotateY(${dx*7}deg) translateY(-6px)`;
        card.style.setProperty('--mx', `${((e.clientX-r.left)/r.width*100).toFixed(1)}%`);
        card.style.setProperty('--my', `${((e.clientY-r.top)/r.height*100).toFixed(1)}%`);
      });
      card.addEventListener('mouseleave', () => { card.style.transform = ''; });
    });
  }
  new MutationObserver(setupTilt).observe(document.body, {childList:true, subtree:true});
  setupTilt();
})();
</script>
"""
    try:
        st.html(anim_html)
    except AttributeError:
        pass   # older Streamlit — aurora CSS still works


def add_timeline(title: str, desc: str):
    st.session_state.timeline.append({
        "title": title, "desc": desc,
        "time": datetime.datetime.now().strftime("%H:%M:%S"), "active": True
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
                "city": city.title(), "temp": d["main"]["temp"],
                "feels_like": d["main"]["feels_like"],
                "humidity": d["main"]["humidity"], "pressure": d["main"]["pressure"],
                "wind_speed": d["wind"]["speed"],
                "description": d["weather"][0]["description"].title(),
                "icon_code": d["weather"][0]["icon"]
            }
    except Exception:
        pass
    return None

def weather_icon_from_code(code: str) -> str:
    mapping = {
        "01": "☀️", "02": "⛅", "03": "☁️", "04": "☁️",
        "09": "🌧️", "10": "🌦️", "11": "⛈️", "13": "❄️", "50": "🌫️"
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
        elif isinstance(msg, AIMessage) and msg.content:
            lines.append(f"🤖 CityMind:\n{msg.content}\n")
        elif isinstance(msg, ToolMessage):
            lines.append(f"🔧 Tool Result:\n{msg.content}\n")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 6. SESSION STATE
# ═══════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "messages": [],
        "agent_state": "idle",
        "pending_tool_call": None,
        "timeline": [],
        "stop_requested": False,
        "theme": "dark",
        "last_weather_city": None,
        "current_action": "Waiting for input...",
        "auto_approve": True,
        "approved_tools": set(),
        # Model selection — default to best free Mistral model
        "provider_key": DEFAULT_PROVIDER,
        # Throttle timestamp — track last API call time
        "last_llm_call_time": 0.0,
        "metrics": {
            "total_messages": 0, "tool_calls": 0,
            "weather_requests": 0, "news_searches": 0,
            "response_times": [], "start_time": time.time()
        },
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ═══════════════════════════════════════════════════════════════
# 7. AGENT EXECUTION LOGIC
#
# FIX 1 — Mistral 400 (mismatched tool calls):
#   build_clean_messages() sanitises the history before every call.
#
# FIX 2 — Repeated approval prompts:
#   If auto_approve is on OR the tool was already approved this session,
#   skip the approval_pending state and go straight to executing.
# ═══════════════════════════════════════════════════════════════

def build_clean_messages(messages: list) -> list:
    """
    Return a message list safe for Mistral:
    • Every AIMessage with tool_calls must be immediately followed by
      one ToolMessage per tool_call id.
    • Orphaned tool-call AIMessages (no matching ToolMessage) are dropped.
    """
    clean = [
        SystemMessage(content=(
            "You are CityMind AI, a helpful and professional agent. "
            "CRITICAL INSTRUCTION: When answering news queries, you must ALWAYS include the exact URL links for every news article provided in the tool results. "
            "Format them as clickable markdown links, e.g., [Article Title](URL)."
        ))
    ]
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            # collect following ToolMessages
            j = i + 1
            tool_responses = {}
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                tool_responses[messages[j].tool_call_id] = messages[j]
                j += 1
            expected = {tc["id"] for tc in msg.tool_calls}
            if expected.issubset(tool_responses.keys()):
                clean.append(msg)
                for tc_id in expected:
                    clean.append(tool_responses[tc_id])
                i = j
            else:
                # orphaned — drop this and everything after
                break
        else:
            if not isinstance(msg, ToolMessage):
                clean.append(msg)
            i += 1
    return clean


def should_auto_approve(tool_name: str) -> bool:
    """Return True if we can skip the human approval step."""
    if st.session_state.get("auto_approve", False):
        return True
    if tool_name in st.session_state.get("approved_tools", set()):
        return True
    return False


def _call_llm_with_retry(msgs, max_retries: int = 4):
    """Call LLM with throttle + response cache + exponential backoff on 429."""
    provider_key = st.session_state.get("provider_key", "mistral_large")
    llm_instance  = get_active_llm()

    # ── 1. Check response cache ────────────────────────────────
    ck = _cache_key(msgs)
    if ck and ck in _RESPONSE_CACHE:
        st.toast("⚡ Answered from cache (no API call)", icon="💾")
        return _RESPONSE_CACHE[ck]

    # ── 2. Throttle — enforce minimum gap between calls ────────
    gap = MIN_CALL_GAP.get(provider_key, 2.0)
    last_call = st.session_state.get("last_llm_call_time", 0)
    elapsed_since = time.time() - last_call
    if elapsed_since < gap:
        time.sleep(gap - elapsed_since)

    # ── 3. Call with exponential backoff on 429 ────────────────
    base_delay = 5
    for attempt in range(max_retries):
        try:
            st.session_state["last_llm_call_time"] = time.time()
            result = llm_instance.invoke(msgs)

            # Cache non-tool-call responses only
            if ck and not getattr(result, "tool_calls", None):
                if len(_RESPONSE_CACHE) >= CACHE_MAX:
                    oldest = next(iter(_RESPONSE_CACHE))
                    del _RESPONSE_CACHE[oldest]
                _RESPONSE_CACHE[ck] = result

            return result

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in str(e) or "rate limit" in err_str or "rate_limited" in err_str
            is_last = attempt == max_retries - 1

            if is_rate_limit and not is_last:
                wait_secs = base_delay * (2 ** attempt)  # 5 → 10 → 20 → 40s
                st.toast(
                    f"⏳ Rate limit — retrying in {wait_secs}s "
                    f"(attempt {attempt+1}/{max_retries-1})…",
                    icon="⚠️"
                )
                time.sleep(wait_secs)
                continue
            raise



def do_thinking_step():
    if st.session_state.stop_requested:
        st.session_state.agent_state = "idle"
        st.session_state.stop_requested = False
        st.toast("⏹ Generation stopped", icon="⏹")
        return

    st.session_state.current_action = "🧠 AI is thinking..."
    add_timeline("AI Thinking", "Processing with Mistral Large...")

    try:
        t0 = time.time()
        clean_msgs = build_clean_messages(st.session_state.messages)
        response = _call_llm_with_retry(clean_msgs)
        elapsed = time.time() - t0
        st.session_state.metrics["response_times"].append(elapsed)
        st.session_state.messages.append(response)

        if getattr(response, "tool_calls", None):
            tc = response.tool_calls[0]
            args = tc.get("args", {})
            if isinstance(args, str):
                args = json.loads(args)
            tc["args"] = args
            st.session_state.pending_tool_call = tc

            # ── FIX: skip approval if auto_approve or already trusted ──
            if should_auto_approve(tc["name"]):
                st.session_state.agent_state = "executing"
                st.session_state.current_action = f"⚡ Auto-executing {tc['name']}..."
                add_timeline("Auto-Execute", f"{tc['name']}({args})")
            else:
                st.session_state.agent_state = "approval_pending"
                st.session_state.current_action = f"⏳ Awaiting approval for {tc['name']}"
                add_timeline("Tool Selected", f"{tc['name']}({args})")
        else:
            st.session_state.agent_state = "idle"
            st.session_state.current_action = "✅ Response ready"
            add_timeline("Final Answer", "AI generated the response")

    except Exception as e:
        err_str = str(e)
        is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
        if is_rate_limit:
            friendly = (
                "⚠️ **Mistral API rate limit reached.** "
                "Please wait a minute then try again, or upgrade your Mistral plan for higher limits."
            )
        else:
            friendly = f"❌ **Error:** {err_str}"
        st.session_state.messages.append(AIMessage(content=friendly))
        st.session_state.agent_state = "idle"
        st.session_state.current_action = "❌ Error occurred"
        label = "Rate Limited" if is_rate_limit else "Error"
        add_timeline(label, err_str[:80])
        st.toast("⚠️ Rate limit hit — please wait a moment" if is_rate_limit else "An error occurred", icon="⚠️")



def do_executing_step():
    tc = st.session_state.pending_tool_call
    tool_name = tc["name"]
    tool_args = tc.get("args", {})
    tool_id   = tc["id"]

    st.session_state.current_action = f"⚡ Executing {tool_name}..."
    add_timeline("Tool Executed", f"Running {tool_name}...")

    try:
        result = TOOLS_MAP[tool_name].invoke(tool_args)
        st.session_state.messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)
        )
        st.session_state.metrics["tool_calls"] += 1
        if tool_name == "get_weather":
            st.session_state.metrics["weather_requests"] += 1
            st.session_state.last_weather_city = tool_args.get("city", "")
        elif tool_name == "get_news":
            st.session_state.metrics["news_searches"] += 1
        add_timeline("Result Received", "Tool returned data successfully")
    except Exception as e:
        st.session_state.messages.append(
            ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_id, name=tool_name)
        )
        add_timeline("Tool Error", str(e))

    st.session_state.pending_tool_call = None
    st.session_state.agent_state = "thinking"


# ═══════════════════════════════════════════════════════════════
# 8. UI RENDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:

        # ══════════════════════════════════════════════
        # HEADER — Logo + Branding
        # ══════════════════════════════════════════════
        state  = st.session_state.agent_state
        status_cfg = {
            "idle":             ("#22C55E", "Online",           "●"),
            "thinking":         ("#F59E0B", "Thinking...",      "◉"),
            "approval_pending": ("#F97316", "Awaiting Input",   "◎"),
            "executing":        ("#06B6D4", "Executing Tool",   "◉"),
        }
        s_color, s_label, s_dot = status_cfg.get(state, ("#22C55E", "Online", "●"))

        st.markdown(f"""
<style>
/* ── Professional Sidebar ────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {{
    background: linear-gradient(180deg,
        rgba(10,15,35,0.98) 0%,
        rgba(7,12,28,1) 100%) !important;
    border-right: 1px solid rgba(124,58,237,0.18) !important;
    padding: 0 !important;
}}
[data-testid="stSidebar"] {{ min-width:280px !important; max-width:280px !important; }}

/* ── section headings ── */
.sb-heading {{
    font-size: .65rem; font-weight: 700; letter-spacing: 1.4px;
    text-transform: uppercase; color: #475569;
    margin: 22px 0 8px; padding: 0 4px;
}}

/* ── stat tile ── */
.sb-stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 6px; }}
.sb-stat {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 12px 10px;
    text-align: center;
}}
.sb-stat-val {{
    font-size: 1.3rem; font-weight: 800; color: #EDF2FF; line-height: 1;
}}
.sb-stat-lbl {{
    font-size: .62rem; color: #64748B; margin-top: 4px;
    text-transform: uppercase; letter-spacing: .8px; font-weight: 600;
}}

/* ── tool badge row ── */
.sb-tool {{
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px; border-radius: 10px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 6px;
}}
.sb-tool-dot {{
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
}}
.sb-tool-name {{ font-size: .78rem; font-weight: 600; color: #CBD5E1; flex: 1; }}
.sb-tool-badge {{
    font-size: .6rem; font-weight: 700; padding: 2px 7px;
    border-radius: 20px; letter-spacing: .5px;
}}

/* ── model card ── */
.sb-model-card {{
    background: rgba(124,58,237,0.08);
    border: 1px solid rgba(124,58,237,0.22);
    border-radius: 14px; padding: 14px 14px 12px;
    margin-bottom: 4px;
}}
.sb-model-name {{
    font-size: .82rem; font-weight: 700; color: #C4B5FD;
    margin-bottom: 4px;
}}
.sb-model-id {{
    font-size: .68rem; color: #6366F1; font-family: monospace;
}}
.sb-model-footer {{
    display: flex; align-items: center; justify-content: space-between;
    margin-top: 10px;
}}
.sb-model-chip {{
    font-size: .58rem; font-weight: 700; letter-spacing: .6px;
    padding: 2px 8px; border-radius: 20px;
    text-transform: uppercase;
}}
.chip-free {{ background: rgba(34,197,94,0.15); color: #22C55E; border: 1px solid rgba(34,197,94,0.3); }}
.chip-paid {{ background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }}

/* ── status card ── */
.sb-status-card {{
    display: flex; align-items: center; gap: 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 12px 14px;
}}
.sb-status-indicator {{
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}}
.sb-status-text {{ font-size: .82rem; font-weight: 700; color: #EDF2FF; }}
.sb-status-sub  {{ font-size: .68rem; color: #64748B; margin-top: 2px; }}

/* ── divider ── */
.sb-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
    margin: 18px 0;
}}

/* ── action buttons ── */
section[data-testid="stSidebar"] .stButton > button {{
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: .78rem !important;
    padding: 8px 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.05) !important;
    color: #CBD5E1 !important;
    transition: all .2s !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(124,58,237,0.2) !important;
    border-color: rgba(124,58,237,0.4) !important;
    color: #EDF2FF !important;
    transform: translateY(-1px) !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(124,58,237,0.25) !important;
    border-radius: 10px !important;
    font-size: .8rem !important;
}}
section[data-testid="stSidebar"] .stToggle > label {{
    font-size: .78rem !important; font-weight: 600 !important;
}}
</style>

<div style="padding: 24px 18px 0;">

  <!-- Avatar + Title -->
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
    <div style="
      width:46px;height:46px;border-radius:14px;flex-shrink:0;
      background:linear-gradient(135deg,#7C3AED,#06B6D4);
      display:flex;align-items:center;justify-content:center;
      font-size:1.5rem;
      box-shadow:0 0 20px rgba(124,58,237,0.5);
    ">🌍</div>
    <div>
      <div style="font-size:1rem;font-weight:800;
        background:linear-gradient(135deg,#EDF2FF,#C4B5FD);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;letter-spacing:-.02em;line-height:1.2;">
        CityMind AI
      </div>
      <div style="font-size:.65rem;color:#475569;margin-top:3px;font-weight:600;
        letter-spacing:.5px;text-transform:uppercase;">
        Intelligent Agent v2.0
      </div>
    </div>
  </div>

  <!-- Status Card -->
  <div class="sb-status-card">
    <div class="sb-status-indicator"
      style="background:rgba({','.join(str(int(s_color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15);">
      <span style="color:{s_color};animation:{'soft-pulse 1.5s infinite' if state != 'idle' else 'none'};">{s_dot}</span>
    </div>
    <div>
      <div class="sb-status-text" style="color:{s_color};">{s_label}</div>
      <div class="sb-status-sub">{st.session_state.get('current_action','Waiting...')[:42]}</div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

        # ── MODEL SELECTOR ─────────────────────────────────────────
        st.markdown('<div style="padding:0 18px;"><div class="sb-heading">🤖 Language Model</div></div>',
                    unsafe_allow_html=True)

        provider_keys   = list(PROVIDER_CATALOGUE.keys())
        provider_labels = [PROVIDER_CATALOGUE[k][0] for k in provider_keys]
        current_key     = st.session_state.get("provider_key", DEFAULT_PROVIDER)
        current_idx     = provider_keys.index(current_key) if current_key in provider_keys else 0

        with st.container():
            chosen_label = st.selectbox(
                "Select Model",
                options=provider_labels,
                index=current_idx,
                key="model_selectbox",
                label_visibility="collapsed",
            )
        chosen_key = provider_keys[provider_labels.index(chosen_label)]
        if chosen_key != st.session_state.get("provider_key"):
            st.session_state["provider_key"] = chosen_key
            _RESPONSE_CACHE.clear()
            st.toast(f"Switched to {PROVIDER_CATALOGUE[chosen_key][1]}", icon="🤖")
            st.rerun()

        active_model_id = PROVIDER_CATALOGUE[chosen_key][1]
        gap = MIN_CALL_GAP.get(chosen_key, 2.0)
        is_paid = "PAID" in PROVIDER_CATALOGUE[chosen_key][0]
        chip_cls = "chip-paid" if is_paid else "chip-free"
        chip_lbl = "PAID" if is_paid else "FREE"

        st.markdown(f"""
<div style="padding: 0 18px 0;">
<div class="sb-model-card">
  <div class="sb-model-name">{active_model_id.replace('open-','').replace('-latest','').replace('-',' ').title()}</div>
  <div class="sb-model-id">{active_model_id}</div>
  <div class="sb-model-footer">
    <span class="sb-model-chip {chip_cls}">{chip_lbl}</span>
    <span style="font-size:.65rem;color:#475569;">⏱ {gap}s throttle</span>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

        # ── SESSION STATS ───────────────────────────────────────────
        m = st.session_state.metrics
        avg_rt = (sum(m["response_times"]) / len(m["response_times"])) if m["response_times"] else 0
        dur = time.time() - m["start_time"]

        st.markdown("""<div style="padding:0 18px;">
<div class="sb-heading">📊 Session Stats</div>""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="sb-stat-grid">
  <div class="sb-stat">
    <div class="sb-stat-val">{m['total_messages']}</div>
    <div class="sb-stat-lbl">Messages</div>
  </div>
  <div class="sb-stat">
    <div class="sb-stat-val">{m['tool_calls']}</div>
    <div class="sb-stat-lbl">Tool Calls</div>
  </div>
  <div class="sb-stat">
    <div class="sb-stat-val">{avg_rt:.1f}s</div>
    <div class="sb-stat-lbl">Avg. Speed</div>
  </div>
  <div class="sb-stat">
    <div class="sb-stat-val">{format_duration(dur)}</div>
    <div class="sb-stat-lbl">Uptime</div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

        # ── TOOLS REGISTRY ──────────────────────────────────────────
        st.markdown("""<div style="padding:0 18px;">
<div class="sb-divider"></div>
<div class="sb-heading">🔧 Tool Registry</div>

<div class="sb-tool">
  <div class="sb-tool-dot" style="background:#06B6D4;box-shadow:0 0 6px #06B6D4;"></div>
  <span class="sb-tool-name">🌦️ Weather API</span>
  <span class="sb-tool-badge" style="background:rgba(6,182,212,0.12);color:#06B6D4;border:1px solid rgba(6,182,212,0.25);">LIVE</span>
</div>

<div class="sb-tool">
  <div class="sb-tool-dot" style="background:#22C55E;box-shadow:0 0 6px #22C55E;"></div>
  <span class="sb-tool-name">📰 Tavily Search</span>
  <span class="sb-tool-badge" style="background:rgba(34,197,94,0.12);color:#22C55E;border:1px solid rgba(34,197,94,0.25);">LIVE</span>
</div>

<div class="sb-tool">
  <div class="sb-tool-dot" style="background:#A78BFA;box-shadow:0 0 6px #A78BFA;"></div>
  <span class="sb-tool-name">🔐 Approval Guard</span>
  <span class="sb-tool-badge" style="background:rgba(167,139,250,0.12);color:#A78BFA;border:1px solid rgba(167,139,250,0.25);">ACTIVE</span>
</div>

<div class="sb-tool">
  <div class="sb-tool-dot" style="background:#F59E0B;box-shadow:0 0 6px #F59E0B;"></div>
  <span class="sb-tool-name">💾 Response Cache</span>
  <span class="sb-tool-badge" style="background:rgba(245,158,11,0.12);color:#F59E0B;border:1px solid rgba(245,158,11,0.25);">ON</span>
</div>

<div class="sb-divider"></div>
<div class="sb-heading">⚙️ Settings</div>
</div>""", unsafe_allow_html=True)

        # ── AUTO-APPROVE TOGGLE ─────────────────────────────────────
        auto = st.toggle(
            "⚡ Auto-Approve All Tools",
            value=st.session_state.auto_approve,
            key="toggle_auto_approve",
            help="When ON, tools run instantly without asking for approval each time."
        )
        if auto != st.session_state.auto_approve:
            st.session_state.auto_approve = auto
            st.rerun()

        if not st.session_state.auto_approve:
            approved = st.session_state.get("approved_tools", set())
            if approved:
                st.markdown(
                    f'<div style="font-size:.7rem;color:#22C55E;margin:6px 4px 0;padding:6px 10px;'
                    f'background:rgba(34,197,94,0.08);border-radius:8px;border:1px solid rgba(34,197,94,0.2);">'
                    f'✔ Trusted: {", ".join(approved)}</div>',
                    unsafe_allow_html=True
                )

        st.markdown('<div style="padding:0 18px;"><div class="sb-divider"></div><div class="sb-heading">🚀 Quick Actions</div></div>',
                    unsafe_allow_html=True)

        # ── ACTION BUTTONS ──────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True, key="btn_clear"):
                st.session_state.messages        = []
                st.session_state.timeline        = []
                st.session_state.agent_state     = "idle"
                st.session_state.pending_tool_call = None
                st.session_state.last_weather_city = None
                st.session_state.current_action  = "Waiting for input..."
                st.session_state.approved_tools  = set()
                st.session_state.metrics = {
                    "total_messages": 0, "tool_calls": 0,
                    "weather_requests": 0, "news_searches": 0,
                    "response_times": [], "start_time": time.time()
                }
                st.toast("Chat cleared", icon="🗑️")
                st.rerun()
        with col2:
            if st.session_state.agent_state != "idle":
                if st.button("⏹ Stop", use_container_width=True, key="btn_stop"):
                    st.session_state.stop_requested = True
                    st.rerun()
            else:
                theme_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
                if st.button(theme_label, use_container_width=True, key="btn_theme"):
                    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                    st.rerun()

        if st.session_state.agent_state == "idle":
            if not (st.session_state.agent_state != "idle"):
                theme_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"

        if st.session_state.messages:
            st.download_button(
                "📥 Export Conversation",
                data=get_chat_export(),
                file_name=f"citymind_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="btn_export"
            )

        if not st.session_state.auto_approve:
            if st.session_state.get("approved_tools"):
                if st.button("🔒 Reset Trust", use_container_width=True, key="btn_reset_trust"):
                    st.session_state.approved_tools = set()
                    st.rerun()

        # ── BOTTOM STRIP ────────────────────────────────────────────
        st.markdown("""
<div style="padding:24px 18px 20px;margin-top:auto;">
  <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);margin-bottom:16px;"></div>
  <div style="display:flex;align-items:center;justify-content:space-between;">
    <div style="font-size:.62rem;color:#334155;font-weight:600;">
      🔗 LangChain &nbsp;·&nbsp; 🧠 Mistral AI
    </div>
    <div style="font-size:.62rem;color:#1E293B;font-weight:600;">v2.0</div>
  </div>
</div>
""", unsafe_allow_html=True)





def render_hero():
    if st.session_state.messages:
        return
    st.markdown("""
    <div style="position:relative;text-align:center;padding:60px 20px 40px;overflow:hidden;">
      <div class="particle" style="width:90px;height:90px;top:8%;left:10%;background:radial-gradient(circle,rgba(124,58,237,.2),transparent 70%);animation-delay:0s;"></div>
      <div class="particle" style="width:70px;height:70px;top:60%;left:78%;background:radial-gradient(circle,rgba(6,182,212,.18),transparent 70%);animation-delay:2.5s;"></div>
      <div class="particle" style="width:55px;height:55px;top:30%;left:85%;background:radial-gradient(circle,rgba(34,197,94,.14),transparent 70%);animation-delay:5s;"></div>
      <div class="particle" style="width:40px;height:40px;top:75%;left:18%;background:radial-gradient(circle,rgba(245,158,11,.16),transparent 70%);animation-delay:1.5s;"></div>
      <div class="fade-in">
        <div style="font-size:4.5rem;margin-bottom:14px;filter:drop-shadow(0 0 32px rgba(124,58,237,.5));display:inline-block;animation:particle-float 6s ease-in-out infinite;">🌍</div>
      </div>
      <h1 class="gradient-title fade-in fade-in-d1">CityMind AI Agent</h1>
      <p class="fade-in fade-in-d2" style="font-size:1.05rem;color:#94A3B8;max-width:580px;margin:18px auto 0;line-height:1.8;font-weight:400;">
        Your intelligent AI assistant that searches live news, checks weather,<br>uses tools autonomously, and reasons before answering.
      </p>
    </div>
    """, unsafe_allow_html=True)

    features = [
        ("🌦️", "Weather Tool",   "Real-time weather from OpenWeather"),
        ("📰", "News Search",    "Live news via Tavily AI Search"),
        ("🤖", "AI Agent",       "Powered by Mistral Large LLM"),
        ("⚡", "Auto-Approve",   "Skip repeated tool confirmations"),
    ]
    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(features):
        with cols[i]:
            st.markdown(f"""
            <div class="feature-card fade-in fade-in-d{i+2}">
              <span class="feature-icon">{icon}</span>
              <div class="feature-name">{name}</div>
              <div class="feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Quick-start prompts
    st.markdown("""
    <div class="fade-in fade-in-d5" style="text-align:center;margin-top:32px;margin-bottom:8px;">
      <div style="font-size:.78rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:12px;">Try asking...</div>
    </div>""", unsafe_allow_html=True)

    pcols = st.columns(3)
    prompts = [
        ("🌦️", "What's the weather in Tokyo?"),
        ("📰", "Latest news from London"),
        ("🌍", "Weather and news for Paris"),
    ]
    for i, (icon, prompt) in enumerate(prompts):
        with pcols[i]:
            if st.button(f"{icon} {prompt}", use_container_width=True, key=f"qp_{i}"):
                st.session_state.messages.append(HumanMessage(content=prompt))
                st.session_state.metrics["total_messages"] += 1
                st.session_state.agent_state = "thinking"
                st.session_state.current_action = "🧠 AI is thinking..."
                add_timeline("User Query", prompt[:80])
                st.rerun()


def render_tool_result_card(msg: ToolMessage):
    content = msg.content

    # ── Weather result ───────────────────────────────────────────
    if content.startswith("Weather in"):
        city = content.split("Weather in ")[1].split(":")[0].strip()
        st.markdown(f"""
        <div class="tool-call-card">
          <span class="tool-icon">🌦️</span>
          <span>Weather data received for <strong>{city}</strong></span>
        </div>""", unsafe_allow_html=True)
        detailed = get_detailed_weather(city)
        if detailed:
            icon = weather_icon_from_code(detailed["icon_code"])
            st.markdown(f"""
            <div class="weather-widget">
              <div style="display:flex;align-items:center;gap:22px;margin-bottom:24px;position:relative;z-index:1;">
                <div style="font-size:3.8rem;filter:drop-shadow(0 4px 12px rgba(6,182,212,.3));">{icon}</div>
                <div>
                  <div style="font-size:.95rem;color:#94A3B8;margin-bottom:3px;font-weight:500;">{detailed['city']}</div>
                  <div class="weather-temp">{detailed['temp']:.0f}°C</div>
                  <div style="font-size:.88rem;color:#94A3B8;margin-top:5px;font-weight:500;">{detailed['description']}</div>
                </div>
              </div>
              <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;position:relative;z-index:1;">
                <div class="weather-stat"><div class="weather-stat-label">Feels Like</div><div class="weather-stat-value">{detailed['feels_like']:.0f}°</div></div>
                <div class="weather-stat"><div class="weather-stat-label">Humidity</div><div class="weather-stat-value">{detailed['humidity']}%</div></div>
                <div class="weather-stat"><div class="weather-stat-label">Wind</div><div class="weather-stat-value">{detailed['wind_speed']} m/s</div></div>
                <div class="weather-stat"><div class="weather-stat-label">Pressure</div><div class="weather-stat-value">{detailed['pressure']} hPa</div></div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            with st.expander("🔧 Raw Weather Data", expanded=False):
                st.code(content, language="text")
        return

    # ── News result (new reliable format) ────────────────────────
    if content.startswith("NEWSFEED::"):
        lines = content.split("\n")
        city  = lines[0].replace("NEWSFEED::", "").strip()
        st.markdown(f"""
        <div class="tool-call-card">
          <span class="tool-icon">📰</span>
          <span>Latest news for <strong>{city.title()}</strong></span>
        </div>""", unsafe_allow_html=True)
        for line in lines[1:]:
            if not line.strip():
                continue
            try:
                parts   = line.split("||")
                title   = parts[0].replace("TITLE::", "").strip()
                url     = parts[1].replace("URL::", "").strip()
                snippet = parts[2].replace("SNIPPET::", "").strip() if len(parts) > 2 else ""
                st.markdown(f"""
                <div class="news-card">
                  <div class="news-title">{title}</div>
                  <div class="news-snippet">{snippet}</div>
                  <a href="{url}" target="_blank" class="news-link">🔗 Open Article →</a>
                </div>""", unsafe_allow_html=True)
            except Exception:
                st.markdown(f"<div class='news-snippet'>{line}</div>", unsafe_allow_html=True)
        return

    # ── Legacy news format (just in case) ────────────────────────
    if content.startswith("Latest news in"):
        city  = content.split("Latest news in ")[1].split(":")[0].strip()
        st.markdown(f"""
        <div class="tool-call-card">
          <span class="tool-icon">📰</span>
          <span>News results for <strong>{city}</strong></span>
        </div>""", unsafe_allow_html=True)
        items = content.split("\n\n")[1:]
        for item in items:
            raw_lines = [l for l in item.strip().split("\n") if l.strip()]
            if len(raw_lines) >= 2:
                title   = raw_lines[0].lstrip("- ").strip()
                url     = raw_lines[1].replace("🔗", "").strip()
                snippet = raw_lines[2].replace("📰", "").strip() if len(raw_lines) > 2 else ""
                st.markdown(f"""
                <div class="news-card">
                  <div class="news-title">{title}</div>
                  <div class="news-snippet">{snippet}</div>
                  <a href="{url}" target="_blank" class="news-link">🔗 Open Article →</a>
                </div>""", unsafe_allow_html=True)
        return

    # ── Fallback ─────────────────────────────────────────────────
    with st.expander("🔧 Tool Result", expanded=False):
        st.code(content, language="text")


def render_chat():
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🌍"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        tname   = tc["name"]
                        targs   = tc.get("args", {})
                        if isinstance(targs, str):
                            targs = json.loads(targs)
                        icon     = "🌦️" if tname == "get_weather" else "📰"
                        args_str = ", ".join(f"{k}={v}" for k, v in targs.items())
                        st.markdown(f"""
                        <div class="tool-call-card">
                          <span class="tool-icon">{icon}</span>
                          <span>Calling <strong>{tname}</strong> with <code>{args_str}</code></span>
                        </div>""", unsafe_allow_html=True)

                if msg.content:
                    st.markdown(msg.content)
                    st.markdown("""
                    <div style="margin-top:10px;">
                      <button class="copy-btn"
                        onclick="navigator.clipboard.writeText(this.closest('.stChatMessageContent').innerText);
                        this.textContent='✅ Copied!';setTimeout(()=>this.textContent='📋 Copy',2000);">
                        📋 Copy
                      </button>
                    </div>""", unsafe_allow_html=True)
        elif isinstance(msg, ToolMessage):
            with st.chat_message("assistant", avatar="🔧"):
                render_tool_result_card(msg)

    # state indicators
    state = st.session_state.agent_state
    if state == "thinking":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("""<div class="typing-indicator">
              <div class="typing-dots"><span></span><span></span><span></span></div>
              <span>AI is thinking...</span></div>""", unsafe_allow_html=True)
    elif state == "executing":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("""<div class="typing-indicator">
              <div class="typing-dots"><span></span><span></span><span></span></div>
              <span>⚡ Executing tool...</span></div>""", unsafe_allow_html=True)
    elif state == "approval_pending":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("""<div class="typing-indicator">
              <span style="animation:soft-pulse 1.5s infinite;font-size:1.2rem;">⏳</span>
              <span>Waiting for your approval...</span></div>""", unsafe_allow_html=True)


def render_approval_panel():
    if st.session_state.agent_state != "approval_pending":
        return
    tc = st.session_state.pending_tool_call
    if not tc:
        return

    tool_name    = tc["name"]
    args         = tc.get("args", {})
    if isinstance(args, str):
        args = json.loads(args)
    icon         = "🌦️" if tool_name == "get_weather" else "📰"
    display_name = "Weather Tool" if tool_name == "get_weather" else "Tavily News Search"
    args_html    = "".join(
        f'<div class="monitor-row"><span class="monitor-label">{k}</span>'
        f'<span class="monitor-value" style="color:#06B6D4;">{v}</span></div>'
        for k, v in args.items()
    )

    st.markdown(f"""
    <div class="approval-panel">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:22px;">
        <span style="font-size:2.2rem;filter:drop-shadow(0 0 12px rgba(124,58,237,.5));">{icon}</span>
        <div>
          <div style="font-size:1.1rem;font-weight:800;letter-spacing:-.01em;">Tool Approval Required</div>
          <div style="font-size:.82rem;color:#8FA3BF;margin-top:3px;">The agent wants to use a tool before responding</div>
        </div>
      </div>
      <div style="background:rgba(0,0,0,.18);border-radius:16px;padding:18px;margin-bottom:26px;border:1px solid rgba(255,255,255,.05);">
        <div class="monitor-row" style="padding-bottom:14px;margin-bottom:10px;border-bottom:1px solid rgba(255,255,255,.06);">
          <span class="monitor-label">Tool Requested</span>
          <span class="monitor-value" style="color:#7C3AED;font-size:.9rem;">{display_name}</span>
        </div>
        <div style="font-size:.7rem;color:#8FA3BF;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;font-weight:700;">Arguments</div>
        {args_html}
      </div>
    </div>""", unsafe_allow_html=True)

    col_a, col_b, col_d = st.columns([2, 2, 1])
    with col_a:
        if st.button("✅ Approve Once", use_container_width=True, type="primary", key="btn_approve"):
            # Remember this tool for the session
            st.session_state.approved_tools.add(tool_name)
            st.session_state.agent_state = "executing"
            st.toast(f"✅ {display_name} approved", icon="✅")
            st.rerun()
    with col_b:
        if st.button("⚡ Approve & Trust All", use_container_width=True, key="btn_trust_all"):
            st.session_state.auto_approve = True
            st.session_state.agent_state = "executing"
            st.toast("⚡ Auto-approve enabled!", icon="⚡")
            st.rerun()
    with col_d:
        if st.button("❌ Deny", use_container_width=True, key="btn_deny"):
            deny_msg = ToolMessage(content="Tool call denied by user.", tool_call_id=tc["id"], name=tc["name"])
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
        items_html = "".join(
            f"""<div class="timeline-item{' active' if item.get('active') else ''}">
              <div class="timeline-title">{item['title']}</div>
              <div class="timeline-desc">{item['desc']} · {item['time']}</div>
            </div>"""
            for item in st.session_state.timeline
        )
        st.markdown(f'<div class="timeline-container">{items_html}</div>', unsafe_allow_html=True)


def render_activity_monitor():
    state = st.session_state.agent_state
    state_display = {
        "idle":             ("Idle",             "#94A3B8"),
        "thinking":         ("Processing",        "#F59E0B"),
        "approval_pending": ("Awaiting Approval", "#F97316"),
        "executing":        ("Executing Tool",    "#06B6D4"),
    }
    state_text, state_color = state_display.get(state, ("Idle", "#94A3B8"))
    current_tool = "None" if not st.session_state.pending_tool_call else st.session_state.pending_tool_call["name"]
    m    = st.session_state.metrics
    avg_rt = (sum(m["response_times"]) / len(m["response_times"])) if m["response_times"] else 0
    auto_lbl = "✅ ON" if st.session_state.auto_approve else "❌ OFF"

    with st.expander("📊 Agent Activity Monitor", expanded=False):
        st.markdown(f"""
        <div style="background:rgba(255,255,255,.03);border-radius:16px;padding:22px;border:1px solid rgba(255,255,255,.06);">
          <div class="monitor-row">
            <span class="monitor-label">Agent Status</span>
            <span class="monitor-value" style="color:{state_color};">● {state_text}</span>
          </div>
          <div class="monitor-row">
            <span class="monitor-label">Auto-Approve</span>
            <span class="monitor-value">{auto_lbl}</span>
          </div>
          <div class="monitor-row">
            <span class="monitor-label">Active Tool</span>
            <span class="monitor-value">{current_tool}</span>
          </div>
          <div class="monitor-row">
            <span class="monitor-label">Current Action</span>
            <span class="monitor-value" style="font-size:.78rem;">{st.session_state.get('current_action','Waiting...')}</span>
          </div>
          <div class="monitor-row">
            <span class="monitor-label">Avg Response</span>
            <span class="monitor-value">{avg_rt:.1f}s</span>
          </div>
        </div>""", unsafe_allow_html=True)


def render_metrics_dashboard():
    m = st.session_state.metrics
    avg_rt = (sum(m["response_times"]) / len(m["response_times"])) if m["response_times"] else 0
    dur    = time.time() - m["start_time"]
    metrics = [
        ("💬", m["total_messages"],        "Total Messages"),
        ("⚡", m["tool_calls"],            "Tool Calls"),
        ("🌦️", m["weather_requests"],     "Weather Requests"),
        ("📰", m["news_searches"],         "News Searches"),
        ("⏱️", f"{avg_rt:.1f}s",          "Avg Response"),
        ("🕐", format_duration(dur),       "Session Duration"),
    ]
    with st.expander("📈 Metrics Dashboard", expanded=False):
        cols = st.columns(3)
        for i, (icon, value, label) in enumerate(metrics):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                  <span class="metric-icon">{icon}</span>
                  <div class="metric-value">{value}</div>
                  <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)


def render_footer():
    st.markdown("""
    <div class="footer-section">
      <div style="font-size:.75rem;color:#64748B;font-weight:500;">Powered by</div>
      <div class="footer-tech">
        <span class="footer-badge">🔗 LangChain</span>
        <span class="footer-badge">🧠 Mistral AI</span>
        <span class="footer-badge">🔍 Tavily Search</span>
        <span class="footer-badge">🌤️ OpenWeather API</span>
        <span class="footer-badge">⚡ Streamlit</span>
      </div>
      <div style="font-size:.7rem;color:#4B5563;margin-top:10px;">Built with ❤️ · CityMind AI Agent © 2025</div>
    </div>""", unsafe_allow_html=True)


def render_thinking_overlay():
    """Full-screen premium overlay shown while the AI is thinking or executing."""
    state  = st.session_state.agent_state
    action = st.session_state.get("current_action", "🧠 AI is thinking...")

    if state == "thinking":
        icon       = "🧠"
        title      = "AI is Thinking"
        subtitle   = f"Processing with {PROVIDER_CATALOGUE.get(st.session_state.get('provider_key', DEFAULT_PROVIDER), ('',))[0].split('[')[0].strip()}"
        orbit_col  = "#7C3AED"
        bar_grad   = "linear-gradient(90deg, #7C3AED, #06B6D4, #22C55E)"
    elif state == "executing":
        icon       = "⚡"
        title      = "Executing Tool"
        subtitle   = action.replace("⚡ ", "").replace("Auto-executing ", "Running: ")
        orbit_col  = "#06B6D4"
        bar_grad   = "linear-gradient(90deg, #06B6D4, #22C55E, #F59E0B)"
    else:
        return   # nothing to show

    overlay_html = f"""
<style>
/* ═══════════════════════════════════════════════════════════
   PREMIUM THINKING OVERLAY — Frosted Glass + Gradient + Blur
═══════════════════════════════════════════════════════════ */

#cm-thinking-overlay {{
    position: fixed; inset: 0; z-index: 9999;
    display: flex; align-items: center; justify-content: center;
    pointer-events: none;
}}

/* ── BACKDROP: blur + gradient wash ── */
.cm-backdrop {{
    position: absolute; inset: 0;
    backdrop-filter: blur(28px) saturate(160%) brightness(0.6);
    -webkit-backdrop-filter: blur(28px) saturate(160%) brightness(0.6);
    background:
        radial-gradient(ellipse 70% 60% at 20% 20%, rgba(124,58,237,0.28), transparent 65%),
        radial-gradient(ellipse 60% 70% at 80% 75%, rgba(6,182,212,0.22), transparent 60%),
        radial-gradient(ellipse 50% 50% at 55% 10%, rgba(236,72,153,0.14), transparent 60%),
        rgba(5, 10, 24, 0.70);
    animation: backdrop-drift 8s ease-in-out infinite alternate;
}}
@keyframes backdrop-drift {{
    0%   {{ filter: hue-rotate(0deg);   }}
    50%  {{ filter: hue-rotate(20deg);  }}
    100% {{ filter: hue-rotate(-15deg); }}
}}

/* ── GRADIENT BORDER wrapper ── */
.cm-card-wrap {{
    position: relative; z-index: 1;
    border-radius: 30px;
    padding: 2px;                /* space for the gradient border */
    background: linear-gradient(135deg,
        rgba(124,58,237,0.9),
        rgba(6,182,212,0.7),
        rgba(236,72,153,0.6),
        rgba(34,197,94,0.5));
    background-size: 300% 300%;
    animation: border-rotate 4s linear infinite;
    box-shadow:
        0 0 60px rgba(124,58,237,0.35),
        0 0 120px rgba(6,182,212,0.20),
        0 40px 100px rgba(0,0,0,0.75);
}}
@keyframes border-rotate {{
    0%   {{ background-position: 0%   50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0%   50%; }}
}}
@keyframes card-pop {{
    from {{ transform: scale(.84) translateY(28px); opacity:0; }}
    to   {{ transform: scale(1)   translateY(0);    opacity:1; }}
}}

/* ── FROSTED GLASS CARD ── */
.cm-card {{
    position: relative;
    background: rgba(12, 20, 42, 0.72);
    backdrop-filter: blur(32px) saturate(200%) brightness(1.08);
    -webkit-backdrop-filter: blur(32px) saturate(200%) brightness(1.08);
    border-radius: 28px;
    padding: 52px 60px 44px;
    text-align: center;
    width: 400px;
    max-width: 88vw;
    overflow: hidden;
}}

/* Frosted inner highlight (top-left gleam) */
.cm-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 60%;
    background: linear-gradient(170deg,
        rgba(255,255,255,0.07) 0%,
        transparent 70%);
    border-radius: 28px 28px 0 0;
    pointer-events: none;
}}

/* Animated aurora blobs behind card content */
.cm-card::after {{
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(circle at 20% 30%, rgba(124,58,237,0.18) 0%, transparent 55%),
        radial-gradient(circle at 80% 70%, rgba(6,182,212,0.14) 0%, transparent 50%);
    animation: card-aurora 5s ease-in-out infinite alternate;
    pointer-events: none;
    border-radius: 28px;
}}
@keyframes card-aurora {{
    0%   {{ opacity: 0.6; transform: scale(1);    }}
    100% {{ opacity: 1.0; transform: scale(1.08); }}
}}

/* ── ORB ── */
.cm-orb-wrap {{
    position: relative;
    width: 124px; height: 124px;
    margin: 0 auto 30px;
    z-index: 2;
}}
.cm-orb-core {{
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 3.1rem;
    z-index: 3;
    animation: orb-breathe 2.6s ease-in-out infinite;
}}
@keyframes orb-breathe {{
    0%,100% {{
        transform: scale(1);
        filter: drop-shadow(0 0 18px {orbit_col}) drop-shadow(0 0 40px {orbit_col}88);
    }}
    50% {{
        transform: scale(1.15);
        filter: drop-shadow(0 0 32px {orbit_col}) drop-shadow(0 0 70px {orbit_col}66);
    }}
}}

/* Pulse rings */
.cm-pulse {{
    position: absolute; inset: 8px;
    border-radius: 50%;
    border: 2px solid {orbit_col};
    animation: cm-ring-pulse 2.6s ease-out infinite;
    opacity: 0;
}}
.cm-pulse:nth-child(2) {{ animation-delay: 0.65s; border-color: #06B6D4; }}
.cm-pulse:nth-child(3) {{ animation-delay: 1.3s;  border-color: #22C55E; }}
@keyframes cm-ring-pulse {{
    0%   {{ transform: scale(.55); opacity: .8; }}
    100% {{ transform: scale(2.4); opacity: 0;  }}
}}

/* Orbit rings */
.cm-orbit {{
    position: absolute; inset: -20px;
    border-radius: 50%;
    border: 1.5px solid rgba(124,58,237,0.30);
    animation: cm-spin linear infinite;
}}
.cm-orbit-1 {{ animation-duration: 3.5s; }}
.cm-orbit-2 {{
    animation-duration: 5.5s; animation-direction: reverse;
    inset: -30px; border-color: rgba(6,182,212,0.28);
}}
.cm-orbit-3 {{
    animation-duration: 9s;
    inset: -48px; border-color: rgba(236,72,153,0.18);
}}
@keyframes cm-spin {{ to {{ transform: rotate(360deg); }} }}

.cm-orbit-dot {{
    position: absolute; top: -6px; left: 50%;
    transform: translateX(-50%);
    width: 12px; height: 12px; border-radius: 50%;
    background: {orbit_col};
    box-shadow: 0 0 14px 4px {orbit_col}88;
}}
.cm-orbit-2 .cm-orbit-dot {{
    background: #06B6D4;
    box-shadow: 0 0 14px 4px #06B6D488;
}}
.cm-orbit-3 .cm-orbit-dot {{
    background: #EC4899;
    box-shadow: 0 0 12px 3px #EC489988;
}}

/* ── TEXT ── */
.cm-title {{
    font-family: 'Inter', sans-serif;
    font-size: 1.55rem; font-weight: 900;
    letter-spacing: -.025em;
    background: linear-gradient(135deg, #EDF2FF 30%, {orbit_col} 70%, #06B6D4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
    position: relative; z-index: 2;
}}
.cm-subtitle {{
    font-size: .82rem; color: #94A3B8;
    margin-bottom: 26px; font-weight: 500;
    letter-spacing: .01em;
    position: relative; z-index: 2;
}}

/* ── DOTS ── */
.cm-dots {{
    display: flex; gap: 8px; justify-content: center;
    margin-bottom: 30px; position: relative; z-index: 2;
}}
.cm-dots span {{
    width: 10px; height: 10px; border-radius: 50%;
    background: {orbit_col};
    box-shadow: 0 0 8px {orbit_col};
    animation: cm-dot-bounce 1s ease-in-out infinite;
}}
.cm-dots span:nth-child(2) {{
    animation-delay: .2s;
    background: #06B6D4; box-shadow: 0 0 8px #06B6D4;
}}
.cm-dots span:nth-child(3) {{
    animation-delay: .4s;
    background: #22C55E; box-shadow: 0 0 8px #22C55E;
}}
@keyframes cm-dot-bounce {{
    0%,80%,100% {{ transform: translateY(0);     opacity: .45; }}
    40%          {{ transform: translateY(-14px);  opacity: 1;   }}
}}

/* ── PROGRESS BAR ── */
.cm-bar-track {{
    height: 5px; border-radius: 5px;
    background: rgba(255,255,255,0.07);
    overflow: hidden; position: relative; z-index: 2;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.4);
}}
.cm-bar-fill {{
    height: 100%;
    background: {bar_grad};
    background-size: 300% 100%;
    border-radius: 5px;
    animation: cm-bar-shimmer 2s linear infinite;
    box-shadow: 0 0 12px rgba(124,58,237,0.5);
}}
@keyframes cm-bar-shimmer {{
    0%   {{ background-position: 300% 0; }}
    100% {{ background-position: -300% 0; }}
}}
</style>

<div id="cm-thinking-overlay">
  <div class="cm-backdrop"></div>
  <div class="cm-card-wrap">
    <div class="cm-card">

      <!-- Orb + rings -->
      <div class="cm-orb-wrap">
        <div class="cm-pulse"></div>
        <div class="cm-pulse"></div>
        <div class="cm-pulse"></div>
        <div class="cm-orbit cm-orbit-1"><div class="cm-orbit-dot"></div></div>
        <div class="cm-orbit cm-orbit-2"><div class="cm-orbit-dot"></div></div>
        <div class="cm-orbit cm-orbit-3"><div class="cm-orbit-dot"></div></div>
        <div class="cm-orb-core">{icon}</div>
      </div>

      <div class="cm-title">{title}</div>
      <div class="cm-subtitle">{subtitle}</div>

      <div class="cm-dots">
        <span></span><span></span><span></span>
      </div>

      <div class="cm-bar-track">
        <div class="cm-bar-fill"></div>
      </div>

    </div>
  </div>
</div>
"""

    try:
        st.html(overlay_html)
    except AttributeError:
        pass   # fallback: no overlay on older Streamlit


# ═══════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    init_session_state()
    inject_css(get_css(st.session_state.theme))
    inject_bg_animation()   # canvas particles + mouse spotlight + 3D tilt

    # ── Show premium thinking overlay FIRST (renders on top via fixed pos) ──
    if st.session_state.agent_state in ("thinking", "executing"):
        render_thinking_overlay()

    # State machine
    if st.session_state.agent_state == "thinking":
        do_thinking_step()
        if st.session_state.agent_state != "approval_pending":
            st.rerun()
    elif st.session_state.agent_state == "executing":
        do_executing_step()
        st.rerun()

    render_sidebar()
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
        st.chat_input("Agent is busy...", key="disabled_chat_input", disabled=True)


if __name__ == "__main__":
    main()