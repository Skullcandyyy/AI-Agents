import streamlit as st
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from .weather_card import render_weather_card
from .news_card import render_news_card

def render_chat_history():
    for msg in st.session_state.messages:
        if isinstance(msg, SystemMessage):
            continue
            
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        
        # Don't natively render tool messages, they are rendered as cards
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if data.get("type") == "weather":
                    render_weather_card(msg.content)
                elif data.get("type") == "news":
                    render_news_card(msg.content)
                else:
                    st.info(f"Tool executed: {msg.content}")
            except:
                # If not JSON, just show it
                st.info(f"Tool Result: {msg.content}")
            continue

        # AI Message with Tool Calls pending
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # We don't render the raw AIMessage if it's just a tool call
            if not msg.content:
                continue

        avatar = "assets/logo/bot_avatar.svg" if role == "assistant" else None
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.content)
            
            # Add action buttons for assistant messages
            if role == "assistant" and msg.content:
                st.html('''
                <div style="display:flex; gap: 10px; margin-top: 10px; opacity: 0.5;">
                    <span style="cursor:pointer;" title="Copy">📋</span>
                    <span style="cursor:pointer;" title="Like">👍</span>
                    <span style="cursor:pointer;" title="Dislike">👎</span>
                </div>
                ''')

def handle_chat_input(llm_with_tools):
    prompt = st.chat_input("Ask anything about any city...")
    
    if "prefill_prompt" in st.session_state:
        prompt = st.session_state.prefill_prompt
        del st.session_state.prefill_prompt
        
    if prompt:
        st.session_state.stats["total_messages"] += 1
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant", avatar="assets/logo/bot_avatar.svg"):
            with st.spinner("Thinking..."):
                response = llm_with_tools.invoke(st.session_state.messages)
                
                # Check for tool calls
                if response.tool_calls:
                    st.session_state.messages.append(response)
                    st.session_state.pending_tool_calls = response.tool_calls
                    st.rerun()
                else:
                    # Token-by-token streaming effect
                    import time
                    def stream_text():
                        for word in response.content.split(" "):
                            yield word + " "
                            time.sleep(0.02)
                    st.write_stream(stream_text)
                    st.session_state.messages.append(response)
                    st.session_state.stats["total_messages"] += 1
