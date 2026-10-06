import os
from mcp.server.mcpserver import MCPServer
from pipeline import list_documents, retrieve_chunks, generate_summary
from auth import decode_access_token

ACCESS_TOKEN = os.getenv("DOCQUERY_ACCESS_TOKEN")
if not ACCESS_TOKEN:
    raise RuntimeError("Set DOCQUERY_ACCESS_TOKEN get one by logging in via the DocQuery app.")

_payload = decode_access_token(ACCESS_TOKEN)
if not _payload:
    raise RuntimeError("DOCQUERY_ACCESS_TOKEN is invalid or expired. Log in again to get a fresh one.")

USER_ID = _payload["user_id"]

mcp = MCPServer("DocQuery Knowledge Base")


@mcp.tool()
def list_my_documents() -> str:
    """List the documents currently in this user's knowledge base."""
    docs = list_documents(USER_ID)
    return ", ".join(docs) if docs else "No documents uploaded yet."


@mcp.tool()
def search_documents(query: str, source_file: str | None = None) -> str:
    """Search the user's uploaded documents for information relevant to a query.
    Optionally restrict the search to one document by its exact filename."""
    results = retrieve_chunks(query, USER_ID, top_k=8, source_file=source_file)
    if not results:
        return "No relevant results found."
    return "\n\n".join(f"[{r[1]}] {r[2]}" for r in results)


@mcp.tool()
def summarize_document(source_file: str) -> str:
    """Summarize one of the user's uploaded documents by its exact filename."""
    return generate_summary(source_file, USER_ID)


if __name__ == "__main__":
    mcp.run()