import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# Launch the sibling MCP server with the same Python environment as this client.
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).with_name("mcp_server.py"))],
    # MCP stdio launches with a restricted default environment; explicitly pass
    # this opt-in flag so the child server can apply its temporary TLS setting.
    env={
        "WIKIPEDIA_DISABLE_TLS_VERIFY": os.getenv(
            "WIKIPEDIA_DISABLE_TLS_VERIFY", "0"
        )
    },
)

# LangGraph state definition
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    wikipedia_data: str


async def create_graph(session):
    # Load tools from MCP server
    tools = await load_mcp_tools(session)
    info_tool = next(
        (tool for tool in tools if tool.name == "fetch_wikipedia_info"), None
    )
    if info_tool is None:
        raise RuntimeError("MCP server did not provide fetch_wikipedia_info.")

    # Read credentials from the environment; never keep API keys in source code.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY before starting the client."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)

    # Prompt template with user/assistant chat only.
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a research assistant. A Wikipedia lookup was performed for this "
            "turn. Answer the user's question using the retrieved data below, be clear "
            "about what the source says, and cite its article URL. Do not claim that "
            "Wikipedia is unavailable when lookup data is present. If the returned "
            "summary does not contain enough detail, use the available section tools.",
        ),
        ("system", "Wikipedia lookup data for this question:\n{wikipedia_data}"),
        MessagesPlaceholder("messages")
    ])
    chat_llm = prompt_template | llm_with_tools

    async def wikipedia_node(state: State) -> dict:
        user_message = next(
            (message for message in reversed(state["messages"])
             if isinstance(message, HumanMessage)),
            None,
        )
        if user_message is None:
            raise RuntimeError("No user question was provided for Wikipedia search.")

        query = _format_content(user_message.content)
        result = await info_tool.ainvoke({"query": query})
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(f"Wikipedia lookup failed: {result['error']}")

        if isinstance(result, dict):
            wikipedia_data = json.dumps(result, ensure_ascii=False)
        else:
            wikipedia_data = str(result)
        logger.info("Wikipedia lookup completed for query: %s", query)
        return {"wikipedia_data": wikipedia_data}

    async def chat_node(state: State) -> dict:
        response = await chat_llm.ainvoke({
            "messages": state["messages"],
            "wikipedia_data": state["wikipedia_data"],
        })
        if response.tool_calls:
            logger.info(
                "Model requested Wikipedia tools: %s",
                ", ".join(call["name"] for call in response.tool_calls),
            )
        else:
            logger.info("Model answered without requesting an MCP tool")
        return {"messages": [response]}

    # Build LangGraph with tool routing
    graph = StateGraph(State)
    graph.add_node("wikipedia_node", wikipedia_node)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "wikipedia_node")
    graph.add_edge("wikipedia_node", "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())


def _format_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "".join(parts)
    return str(content)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            agent = await create_graph(session)
            print("Wikipedia MCP agent is ready.")

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    break
                if not user_input:
                    continue

                try:
                    response = await agent.ainvoke(
                        {"messages": [("user", user_input)]},
                        config={"configurable": {"thread_id": "wiki-session"}},
                    )
                    print("AI:", _format_content(response["messages"][-1].content))
                except Exception:
                    logger.exception("Error handling the request")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye.")