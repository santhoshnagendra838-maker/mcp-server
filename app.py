import logging
import os
from uuid import uuid4

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from mcp_client import _format_content
from web_agent import get_agent_runner

logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Wikipedia AI",
    page_icon="📚",
    layout="centered",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f7f9ff 0%, #ffffff 360px); }
      [data-testid="stHeader"] { background: transparent; }
      .hero { padding: 1.6rem 0 0.6rem; }
      .hero h1 { letter-spacing: -0.04em; margin-bottom: .35rem; }
      .hero p { color: #64748b; font-size: 1.05rem; }
      [data-testid="stChatMessage"] {
        border: 1px solid #e8edf5; border-radius: 18px;
        padding: .8rem 1rem; background: rgba(255,255,255,.92);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 📚 Wikipedia AI")
    st.caption("OpenAI chat connected to your local Wikipedia MCP server.")
    st.markdown(f"**Model**  \n`{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}`")
    if os.getenv("WIKIPEDIA_DISABLE_TLS_VERIFY") == "1":
        st.warning("Wikipedia TLS verification is disabled for this session.")
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid4())
        st.rerun()
    st.divider()
    st.caption("Your API key stays in the server environment; it is not shown in this UI.")

st.markdown(
    """
    <div class="hero">
      <h1>Ask Wikipedia</h1>
      <p>Ask a question and the assistant can look it up using your MCP tools.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set in the environment running this app.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(_format_content(message.content))
    elif isinstance(message, AIMessage):
        text = _format_content(message.content)
        if text or message.tool_calls:
            with st.chat_message("assistant"):
                if text:
                    st.markdown(text)
                if message.tool_calls:
                    tool_names = ", ".join(call["name"] for call in message.tool_calls)
                    st.caption(f"Using MCP tool: {tool_names}")

prompt = st.chat_input("Ask something about Wikipedia…")
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and thinking…"):
                result = get_agent_runner().invoke(prompt, st.session_state.thread_id)
            st.session_state.messages = result["messages"]
            answer = next(
                message
                for message in reversed(st.session_state.messages)
                if isinstance(message, AIMessage)
            )
            answer_text = _format_content(answer.content)
            if answer_text:
                st.markdown(answer_text)
            elif answer.tool_calls:
                st.info("The assistant requested a tool but did not return a text answer.")
        except Exception as exc:
            logger.exception("Chat request failed")
            st.error(f"Request failed: {exc}")