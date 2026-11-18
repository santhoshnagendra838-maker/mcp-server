# Wikipedia MCP Server and Chat UI

A local MCP server exposing Wikipedia search, article sections, and section content tools. The Streamlit chat UI connects an OpenAI model to those tools through the MCP server.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
```

The API key is read from the server environment. Do not put it in source code or commit it. The model defaults to `gpt-4o-mini`; set `OPENAI_MODEL` to use another OpenAI model.

## Run the browser chat UI

```sh
streamlit run app.py
```

Streamlit opens the local chat interface in a browser. Each submitted message starts the local MCP server over stdio, connects the OpenAI agent to its tools, then returns the answer to the UI.

## Run the terminal client

```sh
python mcp_client.py
```

## Test MCP tools directly

Run the MCP server through the MCP Inspector to call its tools without the OpenAI chat agent:

```sh
npx -y @modelcontextprotocol/inspector .venv/bin/python "$PWD/mcp_server.py"
```
