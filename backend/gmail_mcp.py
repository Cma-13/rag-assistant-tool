"""The Gmail MCP server: a second MCP server the chat's agent can connect to.

DocQuery's own server (docquery_mcp.py) offers document search. This one offers
the user's Gmail, so the agent can do tasks beyond answering from documents:
find emails, read one, and bring a PDF attachment into the knowledge base.

It is mounted in main.py at /gmail-mcp and speaks MCP over Streamable HTTP, like
the DocQuery server. Every call must carry the user's DocQuery login token; the
Gmail account used is the one that user connected (gmail_account.py). Gmail is
only ever read: the permission asked from Google is gmail.readonly.
"""

import base64
import functools
import html
import re
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

import gmail_account
import ingest
from docquery_mcp import user_id_from_request

server = MCPServer(
    "Gmail",
    instructions="Read-only tools over the user's connected Gmail account, "
                 "plus importing a PDF attachment into their DocQuery documents.",
)

MAX_RESULTS = 20       # the most emails one search returns
MAX_BODY_CHARS = 6000  # long emails are cut, so the language model's prompt stays small


# ── Reading Gmail's message format ───────────────────────────────────────────

def _header(message, name):
    for h in message.get("payload", {}).get("headers", []):
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _parts(part):
    """Every part of a message, nested ones included (an email with attachments is a
    tree: multipart/mixed -> multipart/alternative -> text/plain, text/html, file...)."""
    yield part
    for child in part.get("parts", []) or []:
        yield from _parts(child)


def _decode(data):
    # Gmail sends bodies as base64url, sometimes without the "=" padding
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _attachments(message):
    found = []
    for part in _parts(message.get("payload", {})):
        name = part.get("filename")
        if name:
            found.append({
                "filename": name,
                "type": part.get("mimeType", ""),
                "size_kb": round(part.get("body", {}).get("size", 0) / 1024, 1),
            })
    return found


def _html_to_text(text):
    text = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _body_text(message):
    """The readable text of an email: its plain-text part, or its HTML part as text."""
    plain, html_part = None, None
    for part in _parts(message.get("payload", {})):
        if part.get("filename"):
            continue  # an attached file, not the email's text
        data = part.get("body", {}).get("data")
        if not data:
            continue
        mime = part.get("mimeType", "")
        if mime == "text/plain" and plain is None:
            plain = _decode(data).decode("utf-8", errors="replace")
        elif mime == "text/html" and html_part is None:
            html_part = _decode(data).decode("utf-8", errors="replace")
    if plain and plain.strip():
        return re.sub(r"\r\n?", "\n", plain).strip()
    if html_part:
        return _html_to_text(html_part)
    return ""


def _user_zone(ctx):
    """The user's time zone, sent by the DocQuery chat (from their browser), e.g. Asia/Kathmandu."""
    name = (ctx.headers or {}).get("x-user-timezone", "")
    try:
        return ZoneInfo(name) if name else None
    except (ZoneInfoNotFoundError, ValueError):
        return None


def _hour(dt):
    return f"{dt.hour % 12 or 12}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def _friendly_date(message, zone):
    """When the email arrived, in the user's own time: "Today, 4:25 PM", "Yesterday, 9:10 AM",
    "Mon 16 Nov, 9:15 AM" (with the year if it is not this year). Gmail's raw Date header
    ("Mon, 23 Nov 2026 16:25:00 +0545") is in the sender's time zone and hard to read."""
    try:
        dt = datetime.fromtimestamp(int(message["internalDate"]) / 1000, tz=timezone.utc)
    except (KeyError, ValueError, TypeError):
        return _header(message, "Date")
    dt = dt.astimezone(zone) if zone else dt.astimezone()  # no zone sent: the server's own
    today = datetime.now(dt.tzinfo).date()
    if dt.date() == today:
        return f"Today, {_hour(dt)}"
    if dt.date() == today - timedelta(days=1):
        return f"Yesterday, {_hour(dt)}"
    day = f"{dt:%a} {dt.day} {dt:%b}" + (f" {dt.year}" if dt.year != today.year else "")
    return f"{day}, {_hour(dt)}"


def _person(value):
    """'Seema Singdan <seema@gmail.com>' -> ('Seema Singdan', 'seema@gmail.com')"""
    name, address = parseaddr(value or "")
    return (name or address or value or "").strip(), address


def _summary(message, zone=None):
    sender, sender_address = _person(_header(message, "From"))
    return {
        "id": message["id"],
        "from": sender,
        "from_address": sender_address,
        "to": _header(message, "To"),
        "date": _friendly_date(message, zone),
        "subject": _header(message, "Subject") or "(no subject)",
        "snippet": html.unescape(message.get("snippet", "")),
        "attachments": _attachments(message),
    }


def _get_message(user_id, message_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", message_id or ""):
        raise gmail_account.GmailError("That is not a valid email id. Use an id from search_emails.")
    return gmail_account.api_get(user_id, f"/messages/{message_id}", {"format": "full"})


# ── Tools ────────────────────────────────────────────────────────────────────

def _gmail_errors(tool):
    """Gmail problems we expect (not connected, not a PDF, email not found) are passed to
    the model as ToolError, so it can tell the user what went wrong. The MCP server hides
    the message of any other kind of error."""
    @functools.wraps(tool)
    def wrapper(*args, **kwargs):
        try:
            return tool(*args, **kwargs)
        except gmail_account.GmailError as e:
            raise ToolError(str(e)) from e
    return wrapper


_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)


@server.tool(structured_output=True, annotations=_READ_ONLY)
@_gmail_errors
def search_emails(query: str, ctx: Context, max_results: int = 10) -> dict[str, Any]:
    """Search the user's Gmail. `query` uses Gmail search syntax, for example:
    from:ashok@example.com, subject:invoice, has:attachment, filename:pdf,
    newer_than:7d, after:2026/11/01, is:unread - or plain words.
    Returns the newest matching emails with sender, subject, date, a short
    snippet and the names of any attached files."""
    user_id = user_id_from_request(ctx)
    count = max(1, min(int(max_results or 10), MAX_RESULTS))
    listing = gmail_account.api_get(user_id, "/messages", {"q": query, "maxResults": count})
    ids = [m["id"] for m in listing.get("messages", [])[:count]]
    with ThreadPoolExecutor(max_workers=5) as pool:  # fetched side by side, not one after another
        messages = list(pool.map(lambda i: _get_message(user_id, i), ids))
    zone = _user_zone(ctx)
    emails = [_summary(m, zone) for m in messages]
    return {"query": query, "emails": emails,
            "more_available": bool(listing.get("nextPageToken"))}


@server.tool(structured_output=True, annotations=_READ_ONLY)
@_gmail_errors
def read_email(message_id: str, ctx: Context) -> dict[str, Any]:
    """Read one email in full: sender, recipients, date, subject, the text of the
    email and the names of attached files. message_id comes from search_emails."""
    user_id = user_id_from_request(ctx)
    message = _get_message(user_id, message_id)
    body = _body_text(message)
    result = _summary(message, _user_zone(ctx))
    result.pop("snippet")
    result["cc"] = _header(message, "Cc")
    result["body"] = body[:MAX_BODY_CHARS]
    result["body_cut_short"] = len(body) > MAX_BODY_CHARS
    return result


@server.tool(structured_output=True, annotations=ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True))
@_gmail_errors
def import_email_attachment(message_id: str, filename: str, ctx: Context) -> dict[str, Any]:
    """Add a PDF attached to an email to the user's DocQuery documents, so questions
    can be asked about it. It is read and indexed like an uploaded file (this takes
    a few seconds to a minute). Only PDF attachments can be imported. message_id
    and filename come from search_emails or read_email."""
    user_id = user_id_from_request(ctx)
    message = _get_message(user_id, message_id)
    files = [p for p in _parts(message.get("payload", {})) if p.get("filename")]
    wanted = (filename or "").strip().lower()
    part = next((p for p in files if p["filename"].lower() == wanted), None)
    if part is None:
        names = ", ".join(p["filename"] for p in files) or "none"
        raise gmail_account.GmailError(f"This email has no attachment called \"{filename}\". Its attachments: {names}.")
    if not (part["filename"].lower().endswith(".pdf") or part.get("mimeType") == "application/pdf"):
        raise gmail_account.GmailError(f"\"{part['filename']}\" is not a PDF. Only PDF attachments can be imported.")

    body = part.get("body", {})
    if body.get("data"):  # small files can come inside the message itself
        content = _decode(body["data"])
    else:
        attachment = gmail_account.api_get(
            user_id, f"/messages/{message_id}/attachments/{body.get('attachmentId', '')}")
        content = _decode(attachment.get("data", ""))

    try:
        added = ingest.add_pdf(content, part["filename"], user_id)
    except ingest.PdfRejected as e:
        raise gmail_account.GmailError(f"\"{part['filename']}\" could not be imported: {e}")
    return {
        "status": "already in documents" if added["status"] == "done" else "importing",
        "saved_as": added["filename"],
        "job_id": added.get("job_id"),
        "message_id": message_id,
        "from_email": {"subject": _header(message, "Subject"), "from": _header(message, "From")},
    }
