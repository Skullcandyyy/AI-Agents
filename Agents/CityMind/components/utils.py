import os
import requests
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from tavily import TavilyClient

# --- Setup API Keys ---
# Assuming they are loaded via dotenv in app.py

# --- Tools ---
@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    API_KEY = os.getenv("OPEN_WEATHER_API_KEY")
    if not API_KEY:
        return "Error: OPEN_WEATHER_API_KEY not found."
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if str(data.get("cod")) != "200":
            return f"Error : {data.get('message', 'Could not fetch weather data')}"
        
        # We return a structured string that our UI can parse
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        
        # Format as JSON string for easy parsing by UI
        import json
        return json.dumps({
            "type": "weather",
            "city": city,
            "temp": temp,
            "desc": desc,
            "humidity": humidity,
            "wind": wind,
            "raw": f"Weather in {city}:{temp}K,{desc}"
        })
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_news(city: str) -> str:
    """Get latest news of a city"""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "Error: TAVILY_API_KEY not found."
    tavily_client = TavilyClient(api_key=tavily_key)
    try:
        response = tavily_client.search(
            query=f"latest news in {city}",
            search_depth="fast",
            max_results=3
        )
        results = response.get("results", [])
        if not results:
            return f"No news found for {city}."
        
        import json
        return json.dumps({
            "type": "news",
            "city": city,
            "results": results
        })
    except Exception as e:
        return f"Error: {str(e)}"

TOOLS = [get_weather, get_news]
TOOL_MAP = {tool.name: tool for tool in TOOLS}

# --- Initialization ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are CityMind AI, a helpful, premium city assistant. You provide real-time weather and news. Be concise and professional.")
        ]
    if "pending_tool_calls" not in st.session_state:
        st.session_state.pending_tool_calls = None
    if "stats" not in st.session_state:
        st.session_state.stats = {
            "weather_calls": 0,
            "news_calls": 0,
            "total_messages": 0,
            "tokens_approx": 0
        }
    if "api_status" not in st.session_state:
        st.session_state.api_status = {
            "Mistral": bool(os.getenv("MISTRAL_API_KEY")),
            "Weather API": bool(os.getenv("OPEN_WEATHER_API_KEY")),
            "Tavily": bool(os.getenv("TAVILY_API_KEY"))
        }

# --- LLM ---
@st.cache_resource
def get_llm():
    return ChatMistralAI(model="mistral-small-latest", temperature=0.7)

def get_llm_with_tools():
    return get_llm().bind_tools(TOOLS)

