"""DocQuery's own MCP server, running inside the backend.

DocQuery's retrieval (hybrid search over the user's documents) is offered here as
an MCP tool. The chat's agent does not call the search function directly: it
connects to this server as an MCP client (see mcp_client.py), asks it which tools
it has, and calls them over MCP. Other MCP servers can be connected the same way
later, so the agent can do tasks beyond document retrieval.

The server is mounted in main.py at /mcp and speaks MCP over Streamable HTTP.
Every call must carry the user's login token (Authorization: Bearer <JWT>); the
user is always taken from that token, never from what the AI model writes, so a
tool call can only ever see the logged-in user's own documents.
"""

from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

import answer_trace as tr
from auth import decode_access_token
from pipeline import add_following_chunks, list_documents, retrieve_chunks

server = MCPServer(
    "DocQuery",
    instructions="Tools over the user's uploaded PDF documents in DocQuery.",
)


def user_id_from_request(ctx: Context) -> int:
    """The logged-in user, from the request's Authorization header."""
    auth = (ctx.headers or {}).get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    payload = decode_access_token(token) if token else None
    if not payload or "user_id" not in payload:
        # ToolError: the message reaches the caller (other errors are hidden by the MCP server)
        raise ToolError("Not logged in: send a valid DocQuery token in the Authorization header.")
    return payload["user_id"]


def _wants_trace(ctx: Context) -> bool:
    """The DocQuery chat asks for the search steps, to show them under the answer."""
    return (ctx.headers or {}).get("x-docquery-trace") == "1"


@server.tool(structured_output=True, annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False))
def search_documents(
    query: str,
    ctx: Context,
    source_file: str | None = None,
    within: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Search the user's uploaded documents for information relevant to a query.
    Optionally restrict the search to one specific document by its exact filename (source_file)."""
    user_id = user_id_from_request(ctx)
    files = [source_file] if source_file else (within or None)
    if _wants_trace(ctx):
        tr.start()
    try:
        rows = add_following_chunks(retrieve_chunks(query, user_id, top_k=top_k, source_file=files), user_id)
        trace = tr.finish() if tr.active() else None
    finally:
        if tr.active():
            tr.finish()
    return {
        "results": [{"id": r[0], "document": r[1], "text": r[2]} for r in rows],
        "trace": trace,
    }


@server.tool(structured_output=True, annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False))
def list_my_documents(ctx: Context) -> dict[str, Any]:
    """List the names of the documents in the user's knowledge base."""
    return {"documents": list_documents(user_id_from_request(ctx))}
