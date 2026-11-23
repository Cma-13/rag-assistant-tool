"""The chat's MCP client: how the agent reaches its tools.

The agent connects to MCP servers, asks each which tools it offers (list_tools),
gives those to the language model, and when the model picks one, calls it on the
server that offers it (call_tool). Two servers are connected today (SERVERS):
- DocQuery: document search over the user's knowledge base (docquery_mcp.py)
- Gmail: the user's connected Gmail (gmail_mcp.py)
Another server is added by giving it a name and an address in SERVERS.

The agent loop is ordinary (non-async) code running in a worker thread, so each
MCP call runs its own small event loop with asyncio.run.
"""

import asyncio
import contextvars
import json
import os

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp_types import LATEST_PROTOCOL_VERSION

import answer_trace as tr
from auth import create_access_token

# Where the backend reaches the MCP servers. In Docker and in local runs the backend
# listens on port 8000; set these if a server runs somewhere else.
DOCQUERY_MCP_URL = os.getenv("DOCQUERY_MCP_URL", "http://127.0.0.1:8000/mcp/")
GMAIL_MCP_URL = os.getenv("GMAIL_MCP_URL", "http://127.0.0.1:8000/gmail-mcp/")
SERVERS = {
    "DocQuery": DOCQUERY_MCP_URL,
    "Gmail": GMAIL_MCP_URL,
}
TRANSPORT = "Streamable HTTP"

# Our own servers speak the latest MCP version, so the client uses it straight away
# instead of first asking the server which versions it supports (one round trip less).
PROTOCOL_VERSION = LATEST_PROTOCOL_VERSION

# Tool arguments the app fills in itself; the language model never sees them.
_APP_ONLY_ARGS = {"within", "top_k"}

# The user's time zone (from their browser, e.g. "Asia/Kathmandu"), sent to the servers so
# a tool can show times as the user sees them. Set for the question being answered.
user_timezone = contextvars.ContextVar("user_timezone", default="")

_tools_cache = {}   # server name -> its tools, in the language model's format
_tool_server = {}   # tool name -> the server that offers it


class MCPToolError(RuntimeError):
    """The MCP server was reached but the tool reported an error."""


def _headers(user_id, trace):
    # A token for this user, so the server knows whose documents (or Gmail) to use.
    token = create_access_token({"user_id": user_id, "email": "", "via": "mcp"})
    headers = {"Authorization": f"Bearer {token}", "X-DocQuery-Trace": "1" if trace else "0"}
    if user_timezone.get():
        headers["X-User-Timezone"] = user_timezone.get()
    return headers


async def _list_tools(url, user_id):
    async with httpx2.AsyncClient(headers=_headers(user_id, False), timeout=30) as http:
        async with Client(streamable_http_client(url, http_client=http), mode=PROTOCOL_VERSION) as client:
            return (await client.list_tools()).tools


async def _call_tool(url, name, arguments, user_id, trace):
    async with httpx2.AsyncClient(headers=_headers(user_id, trace), timeout=120) as http:
        async with Client(streamable_http_client(url, http_client=http), mode=PROTOCOL_VERSION) as client:
            return await client.call_tool(name, arguments)


def _run(coroutine, url):
    """Run one MCP exchange. Network errors come wrapped in an exception group;
    raise the real one, with a clear message when the server is not reachable."""
    try:
        return asyncio.run(coroutine)
    except BaseExceptionGroup as group:
        first = group
        while isinstance(first, BaseExceptionGroup) and first.exceptions:
            first = first.exceptions[0]
        if isinstance(first, (httpx2.ConnectError, ConnectionError, OSError)):
            raise ConnectionError(f"MCP server not reachable at {url}") from first
        raise first from group


def _for_llm(tool):
    """An MCP tool as an OpenAI-style function definition, without the app-only arguments."""
    schema = dict(tool.input_schema or {"type": "object", "properties": {}})
    props = {k: v for k, v in (schema.get("properties") or {}).items() if k not in _APP_ONLY_ARGS}
    schema = {"type": "object", "properties": props,
              "required": [r for r in schema.get("required", []) if r in props]}
    return {"type": "function", "function": {
        "name": tool.name, "description": tool.description or "", "parameters": schema}}


def tools_for_llm(user_id, servers=("DocQuery",)):
    """The tools of the given MCP servers, in the format the language model expects.
    Each server's list is fetched once, then reused. Raises ConnectionError if a
    server can't be reached."""
    tools = []
    for server in servers:
        if server not in _tools_cache:
            listed = _run(_list_tools(SERVERS[server], user_id), SERVERS[server])
            _tools_cache[server] = [_for_llm(t) for t in listed]
            for t in listed:
                _tool_server[t.name] = server
        tools.extend(_tools_cache[server])
    return tools


def server_of(name):
    """The name of the server that offers a tool (DocQuery's search if unknown)."""
    return _tool_server.get(name, "DocQuery")


def call_tool(name, arguments, user_id):
    """Call a tool on the MCP server that offers it and return its structured result.
    Search steps the server recorded are added to this answer's trace."""
    url = SERVERS[server_of(name)]
    arguments = {k: v for k, v in arguments.items() if v is not None}
    with tr.timer() as t:
        result = _run(_call_tool(url, name, arguments, user_id, tr.active()), url)
    if result.is_error:
        text = " ".join(getattr(c, "text", "") for c in (result.content or []))
        raise MCPToolError(text or f"The MCP tool {name} failed.")
    data = result.structured_content
    if data is None:  # a server that sends its result only as JSON text
        text = "".join(getattr(c, "text", "") for c in (result.content or []))
        try:
            data = json.loads(text) if text else {}
        except ValueError:
            data = {"text": text}
    tr.merge(data.get("trace"))
    return data, t.ms