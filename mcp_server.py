import sys
import os
import logging
from urllib.parse import urlparse

import wikipedia
import wikipedia.wikipedia as wikipedia_api
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolRequest, ListToolsRequest

mcp = FastMCP("WikipediaSearch")
logging.basicConfig(
    level=os.getenv("MCP_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("wikipedia_mcp")


# Wikimedia requires API clients to identify themselves with a descriptive
# User-Agent; the wikipedia package's default is rejected by the API.
wikipedia.set_user_agent(
    "WikipediaSearchMCP/1.0 (Python MCP server; "
    "https://github.com/goldsmith/Wikipedia)"
)

# Temporary local diagnostic only: disable TLS verification for Wikipedia API
# calls when explicitly enabled in the server environment.
if os.getenv("WIKIPEDIA_DISABLE_TLS_VERIFY") == "1":
    _original_requests_get = wikipedia_api.requests.get

    def _requests_get_with_wikipedia_tls_disabled(url, *args, **kwargs):
        if urlparse(url).hostname == "en.wikipedia.org":
            kwargs["verify"] = False
        return _original_requests_get(url, *args, **kwargs)

    wikipedia_api.requests.get = _requests_get_with_wikipedia_tls_disabled
    print(
        "WARNING: TLS verification is disabled for en.wikipedia.org; "
        "do not use this mode for sensitive queries or production.",
        file=sys.stderr,
    )


def _find_wikipedia_page(topic: str):
    results = wikipedia.search(topic)
    if not results:
        return None
    return wikipedia.page(results[0], auto_suggest=False)

@mcp.tool()
def fetch_wikipedia_info(query: str) -> dict:
    """
    Search Wikipedia for a topic and return title, summary, and URL of the best match.
    """
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return {"error": "No results found for your query."}

        best_match = search_results[0]
        # The title already came from Wikipedia search; don't auto-search it again.
        page = wikipedia.page(best_match, auto_suggest=False)

        return {
            "title": page.title,
            "summary": page.summary,
            "url": page.url
        }

    except wikipedia.DisambiguationError as e:
        return {
            "error": f"Ambiguous topic. Try one of these: {', '.join(e.options[:5])}"
        }

    except wikipedia.PageError:
        return {
            "error": "No Wikipedia page could be loaded for this query."
        }

@mcp.tool()
def list_wikipedia_sections(topic: str) -> dict:
    """
    Return a list of section titles from the Wikipedia page of a given topic.
    """
    try:
        page = _find_wikipedia_page(topic)
        if page is None:
            return {"error": f"No Wikipedia page found for '{topic}'."}

        # Wikimedia deprecated parse prop=sections; use the replacement tocdata.
        response = wikipedia_api._wiki_request({
            "action": "parse",
            "prop": "tocdata",
            "pageid": page.pageid,
        })
        sections = [
            section["line"]
            for section in response.get("parse", {})
            .get("tocdata", {})
            .get("sections", [])
        ]
        return {"sections": sections}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_section_content(topic: str, section_title: str) -> dict:
    """
    Return the content of a specific section in a Wikipedia article.
    """
    try:
        page = _find_wikipedia_page(topic)
        if page is None:
            return {"error": f"No Wikipedia page found for '{topic}'."}
        content = page.section(section_title)
        if content:
            return {"content": content}
        else:
            return {"error": f"Section '{section_title}' not found in article '{topic}'."}
    except Exception as e:
        return {"error": str(e)}


def _install_request_logging() -> None:
    handlers = mcp._mcp_server.request_handlers

    list_handler = handlers[ListToolsRequest]

    async def logged_list_handler(request):
        logger.info("Received ListToolsRequest")
        result = await list_handler(request)
        tools = getattr(getattr(result, "root", None), "tools", [])
        logger.info(
            "ListToolsRequest completed; tools=%s",
            ", ".join(tool.name for tool in tools) or "none",
        )
        return result

    call_handler = handlers[CallToolRequest]

    async def logged_call_handler(request):
        name = request.params.name
        arguments = request.params.arguments or {}
        # Log argument names only; values may contain user-provided data.
        logger.info(
            "Received CallToolRequest name=%s argument_keys=%s",
            name,
            ", ".join(sorted(arguments)) or "none",
        )
        result = await call_handler(request)
        call_result = getattr(result, "root", None)
        if getattr(call_result, "isError", False):
            logger.warning("CallToolRequest failed: %s", name)
        else:
            logger.info("CallToolRequest completed: %s", name)
        return result

    handlers[ListToolsRequest] = logged_list_handler
    handlers[CallToolRequest] = logged_call_handler

# Run the MCP server
if __name__ == "__main__":
    _install_request_logging()
    print("Starting MCP Wikipedia Server...", file=sys.stderr)
    mcp.run(transport="stdio")