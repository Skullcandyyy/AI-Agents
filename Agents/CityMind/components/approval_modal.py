import streamlit as st
from langchain_core.messages import ToolMessage
from .utils import TOOL_MAP

@st.dialog("🔒 Permission Required")
def render_approval_modal():
    tool_calls = st.session_state.get("pending_tool_calls", [])
    if not tool_calls:
        st.write("No pending tools.")
        if st.button("Close"):
            st.rerun()
        return

    st.markdown("### The AI wants to execute the following tool(s):")
    
    for tool_call in tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        st.html(f'''
        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid rgba(123, 97, 255, 0.4);">
            <strong style="color: #4F8CFF; font-size: 1.1rem;">🛠️ {name}</strong><br>
            <span style="color: #94A3B8;">Parameters:</span> <code style="background: rgba(0,0,0,0.5); color: #38BDF8;">{args}</code>
        </div>
        ''')

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve", use_container_width=True, type="primary"):
            execute_tools(tool_calls)
            st.session_state.pending_tool_calls = None
            st.rerun()
    with col2:
        if st.button("❌ Reject", use_container_width=True):
            reject_tools(tool_calls)
            st.session_state.pending_tool_calls = None
            st.rerun()

def execute_tools(tool_calls):
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        if tool_name in TOOL_MAP:
            tool_func = TOOL_MAP[tool_name]
            try:
                # Update stats
                if tool_name == "get_weather":
                    st.session_state.stats["weather_calls"] += 1
                elif tool_name == "get_news":
                    st.session_state.stats["news_calls"] += 1
                
                result = tool_func.invoke(tool_args)
                st.session_state.messages.append(ToolMessage(content=result, tool_call_id=tool_id))
            except Exception as e:
                st.session_state.messages.append(ToolMessage(content=f"Error executing {tool_name}: {str(e)}", tool_call_id=tool_id))
        else:
            st.session_state.messages.append(ToolMessage(content=f"Tool {tool_name} not found.", tool_call_id=tool_id))

def reject_tools(tool_calls):
    for tool_call in tool_calls:
        st.session_state.messages.append(ToolMessage(content="User rejected tool execution.", tool_call_id=tool_call["id"]))
