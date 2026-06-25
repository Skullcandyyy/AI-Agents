#1 Loading all the required libraries

from dotenv import load_dotenv
load_dotenv()

import os
import requests 

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage , ToolMessage
from langchain.agents.middleware import wrap_tool_call
from tavily import TavilyClient
from langchain.agents import create_agent
from rich import print

#2 Now create some tools to use with the agent

# Weather tool"

@tool 
def get_weather(city: str)-> str:
    """Get current weather of a city"""
    API_KEY= os.getenv("OPEN_WEATHER_API_KEY")
    url=f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
    response = requests.get(url)
    data=response.json()
    
    if str(data.get("cod")) != "200":
        return f"Error : {data.get('message', 'Could not fetch weather data')}"
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}:{temp}K,{desc}"

#Tavily news tool

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """"Get latest news of a city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="fast",
        max_results=5
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}."
    
    
    news_list=[]
    for r in results:
        title=r.get("title","No title")
        url= r.get("url","")
        snippet= r.get("content","")

        news_list.append(
            f"-{title}\n 🔗 {url}\n  📰 {snippet[:100]}..."

        )

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

llm = ChatMistralAI(model="mistral-large-latest")

@wrap_tool_call
def human_approval(request,handler):
    """Ask for human approval before every tool call."""
    tool_name= request.tool_call["name"]
    confirm= input(f"Agents wants to call '{tool_name}'. Approve? (yes/no):")

    if confirm.lower() != "yes":
        return ToolMessage(
            content= "Tool call denied by user.",
            tool_call_id= request.tool_call["id"]
        )
    
    return handler(request)

agent= create_agent(
    llm,
    tools=[get_news , get_weather],
    system_prompt="you are a helpful city assistant",
    middleware=[human_approval]
)

print("City Agent | type exit to quit")

while True:
    user_input=input("You:")
    if user_input.lower() == "exit":
        break
    result=agent.invoke(
        {"messages": [{"role":"user","content": user_input}]}
    
    )
    print("Bot:", result['messages'][-1].content)