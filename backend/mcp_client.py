"""The chat's MCP client: how the agent reaches its tools.

The agent asks the MCP server which tools it offers (list_tools), gives those to
the language model, and when the model picks one, calls it over MCP (call_tool).
Today the only server is DocQuery's own (docquery_mcp.py, mounted at /mcp);
more servers can be added to SERVERS later.

The agent loop is ordinary (non-async) code running in a worker thread, so each
MCP call runs its own small event loop with asyncio.run.
"""

import asyncio
import json
import os

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

import answer_trace as tr
from auth import create_access_token

# Where the backend reaches its own MCP server. In Docker and in local runs the
# backend listens on port 8000; set DOCQUERY_MCP_URL if you run it on another port.
DOCQUERY_MCP_URL = os.getenv("DOCQUERY_MCP_URL", "http://127.0.0.1:8000/mcp/")

# Tool arguments the app fills in itself; the language model never sees them.
_APP_ONLY_ARGS = {"within", "top_k"}

_tools_cache = None


class MCPToolError(RuntimeError):
    """The MCP server was reached but the tool reported an error."""


def _headers(user_id, trace):
    # A token for this user, so the server knows whose documents to search.
    token = create_access_token({"user_id": user_id, "email": "", "via": "mcp"})
    return {"Authorization": f"Bearer {token}", "X-DocQuery-Trace": "1" if trace else "0"}


async def _list_tools(user_id):
    async with httpx2.AsyncClient(headers=_headers(user_id, False), timeout=30) as http:
        async with Client(streamable_http_client(DOCQUERY_MCP_URL, http_client=http)) as client:
            return (await client.list_tools()).tools


async def _call_tool(name, arguments, user_id, trace):
    async with httpx2.AsyncClient(headers=_headers(user_id, trace), timeout=120) as http:
        async with Client(streamable_http_client(DOCQUERY_MCP_URL, http_client=http)) as client:
            return await client.call_tool(name, arguments)


def _run(coroutine):
    """Run one MCP exchange. Network errors come wrapped in an exception group;
    raise the real one, with a clear message when the server is not reachable."""
    try:
        return asyncio.run(coroutine)
    except BaseExceptionGroup as group:
        first = group
        while isinstance(first, BaseExceptionGroup) and first.exceptions:
            first = first.exceptions[0]
        if isinstance(first, (httpx2.ConnectError, ConnectionError, OSError)):
            raise ConnectionError(f"DocQuery MCP server not reachable at {DOCQUERY_MCP_URL}") from first
        raise first from group


def tools_for_llm(user_id):
    """The MCP server's tools, in the format the language model expects
    (OpenAI-style function definitions). Fetched once, then reused."""
    global _tools_cache
    if _tools_cache is None:
        tools = _run(_list_tools(user_id))
        converted = []
        for tool in tools:
            schema = dict(tool.input_schema or {"type": "object", "properties": {}})
            props = {k: v for k, v in (schema.get("properties") or {}).items() if k not in _APP_ONLY_ARGS}
            schema = {"type": "object", "properties": props,
                      "required": [r for r in schema.get("required", []) if r in props]}
            converted.append({"type": "function", "function": {
                "name": tool.name, "description": tool.description or "", "parameters": schema}})
        _tools_cache = converted
    return _tools_cache


def call_tool(name, arguments, user_id):
    """Call a tool on the MCP server and return its structured result.
    The search steps the server recorded are added to this answer's trace."""
    arguments = {k: v for k, v in arguments.items() if v is not None}
    with tr.timer() as t:
        result = _run(_call_tool(name, arguments, user_id, tr.active()))
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
