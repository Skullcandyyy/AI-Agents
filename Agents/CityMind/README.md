<div align="center">
  <h1>🤖 CityMind AI</h1>
  <p><strong>"Your Intelligent City Assistant"</strong></p>
  <p>Real-time Weather • Live News • AI Agent • Tool Calling • Human Approval</p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/Streamlit-1.34+-FF4B4B.svg" alt="Streamlit">
    <img src="https://img.shields.io/badge/Langchain-0.1+-green.svg" alt="Langchain">
    <img src="https://img.shields.io/badge/Mistral_AI-Powered-orange.svg" alt="Mistral">
  </p>
</div>

<hr>

## 🚀 Overview

**CityMind AI** is a premium, production-ready AI SaaS application built for modern city exploration. Moving beyond basic chatbots, it features an advanced LangChain agent capable of using live tool calling to fetch real-time weather and news. 

Designed with a high-end UI/UX architecture utilizing glassmorphism, animated gradients, and floating components, this application ensures an exceptional user experience right from the first glance.

## ✨ Key Features

- **Premium UI/UX:** Dark theme, glassmorphism cards, animated backgrounds, and smooth CSS transitions.
- **AI Agent with Tools:** Powered by Mistral AI, equipped with OpenWeather API and Tavily Live Search.
- **Human Approval Middleware:** Replaces standard terminal `input()` with a professional in-app Streamlit modal (`@st.dialog`) to ask for human approval before executing any tool.
- **Live Dashboards:** Real-time metrics on tool calls, token usage, and API connectivity status.
- **Rich Output Cards:** Custom HTML/CSS injected components for Weather and News instead of plain markdown.

## 🏗️ Architecture & Structure

```
CityMind/
├── app.py                     # Main application entry point
├── styles.css                 # Premium custom CSS (Animations, Glassmorphism)
├── requirements.txt           # Project dependencies
├── README.md                  # Documentation
├── components/
│   ├── __init__.py
│   ├── analytics.py           # Right panel statistics dashboard
│   ├── approval_modal.py      # Streamlit dialog for tool approval
│   ├── chat.py                # Chat interface and message streaming
│   ├── footer.py              # Custom footer
│   ├── hero.py                # Welcome screen and quick actions
│   ├── news_card.py           # Custom component for rendering news
│   ├── sidebar.py             # Left panel navigation and status
│   ├── utils.py               # Session state and Langchain Tool initialization
│   └── weather_card.py        # Custom component for rendering weather
└── assets/
    ├── animations/
    ├── icons/
    └── logo/
```

## 🛠️ Tech Stack

- **Frontend:** Streamlit, Custom HTML/CSS
- **Backend Core:** Python, LangChain, Requests
- **LLM Engine:** Mistral AI (`mistral-small-latest`)
- **Tools API:** OpenWeather API, Tavily Search API
- **State Management:** Streamlit Session State

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd CityMind
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   MISTRAL_API_KEY=your_mistral_key_here
   OPEN_WEATHER_API_KEY=your_weather_key_here
   TAVILY_API_KEY=your_tavily_key_here
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 🔮 Future Improvements
- [ ] User Authentication (Clerk/Auth0)
- [ ] Chat History Export (PDF/Markdown)
- [ ] Vector Database integration for long-term memory
- [ ] Text-to-Speech (TTS) integration

<hr>

<div align="center">
  <p>Built with ❤️ for AI Engineering Portfolios</p>
</div>
