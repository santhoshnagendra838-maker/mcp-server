import asyncio
import threading
from contextlib import AsyncExitStack
from concurrent.futures import TimeoutError as FutureTimeoutError

from langchain_core.messages import HumanMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from mcp_client import create_graph, server_params


class AgentRunner:
    """Keep async MCP/OpenAI resources on one loop across Streamlit reruns."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="wikipedia-mcp-agent",
            daemon=True,
        )
        self._thread.start()
        self._stack = AsyncExitStack()
        self._agent = None
        self._submit(self._initialize()).result()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    async def _initialize(self):
        read, write = await self._stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._agent = await create_graph(session)

    async def _invoke(self, prompt, thread_id):
        return await self._agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config={"configurable": {"thread_id": thread_id}},
        )

    def invoke(self, prompt, thread_id):
        try:
            return self._submit(self._invoke(prompt, thread_id)).result(timeout=180)
        except FutureTimeoutError as exc:
            raise TimeoutError("The Wikipedia request timed out after 180 seconds.") from exc

    async def _shutdown(self):
        await self._stack.aclose()

    def close(self):
        if self._loop.is_closed():
            return
        self._submit(self._shutdown()).result(timeout=15)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=15)
        self._loop.close()


_runner = None
_runner_lock = threading.Lock()


def get_agent_runner():
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = AgentRunner()
    return _runner
