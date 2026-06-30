import streamlit as st
import json

def render_weather_card(json_str: str):
    try:
        data = json.loads(json_str)
        city = data.get("city", "Unknown")
        temp_k = float(data.get("temp", 0))
        temp_c = int(temp_k - 273.15)
        desc = data.get("desc", "N/A")
        humidity = data.get("humidity", 0)
        wind = data.get("wind", 0)
        
        icon = "☀️"
        weather_class = "weather-sunny"
        
        desc_lower = desc.lower()
        if "thunder" in desc_lower or "storm" in desc_lower:
            icon = "⛈️"
            weather_class = "weather-thunder"
        elif "rain" in desc_lower or "drizzle" in desc_lower:
            icon = "🌧️"
            weather_class = "weather-rain"
        elif "snow" in desc_lower:
            icon = "❄️"
            weather_class = "weather-cloudy"
        elif "cloud" in desc_lower:
            icon = "☁️"
            weather_class = "weather-cloudy"
        elif "clear" in desc_lower:
            icon = "☀️"
            weather_class = "weather-sunny"
        
        st.html(f'''
        <div class="premium-weather-card {weather_class}">
            <div class="weather-animation-layer"></div>
            <div class="weather-main">
                <div>
                    <div class="weather-city">{city}</div>
                    <div class="weather-temp">{temp_c}°C</div>
                    <div class="weather-desc">{icon} {desc}</div>
                </div>
            </div>
            <div class="weather-details">
                <div class="w-detail">
                    <span>Humidity</span>
                    <strong>{humidity}%</strong>
                </div>
                <div class="w-detail">
                    <span>Wind</span>
                    <strong>{wind} m/s</strong>
                </div>
                <div class="w-detail">
                    <span>Status</span>
                    <strong>Live 🟢</strong>
                </div>
            </div>
        </div>
        ''')
    except Exception as e:
        st.error(f"Failed to render weather card: {str(e)}")
