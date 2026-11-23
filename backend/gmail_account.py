"""Connecting a user's Gmail to DocQuery, and calling the Gmail API for them.

How the connection works (Google's OAuth "authorization code" flow):
1. The user clicks "Connect Gmail". The app sends them to Google with connect_url().
2. They pick their Google account and allow read-only access to Gmail.
3. Google sends them back to /gmail/callback with a one-time code; finish_connect()
   swaps it for a refresh token, which is stored encrypted in gmail_accounts.
4. When the Gmail MCP server needs Gmail, api_get() uses the refresh token to get a
   short-lived access token and calls the Gmail API.

Only the gmail.readonly permission is asked for: DocQuery can read emails and
download attachments, but can never send, change or delete anything.
"""

import base64
import hashlib
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import requests
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from jose import JWTError, jwt

from auth import ALGORITHM, SECRET_KEY
from pipeline import db_cursor

load_dotenv()

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GMAIL_REDIRECT_URI", "http://127.0.0.1:8000/gmail/callback")

GMAIL_READONLY = "https://www.googleapis.com/auth/gmail.readonly"
SCOPES = ["openid", "email", GMAIL_READONLY]
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"

# The refresh token is a long-lived key to the user's mailbox, so it is never stored
# as plain text: it is encrypted with a key made from the app's JWT secret.
_fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256((SECRET_KEY + ":gmail").encode()).digest()))

_access_tokens = {}  # user_id -> (access token, time it expires)
_tokens_lock = threading.Lock()
_TABLE_READY = False


class GmailError(Exception):
    """Something went wrong with Gmail; the message can be shown to the user."""


class GmailNotConnected(GmailError):
    """This user has not connected Gmail (or Google no longer accepts the connection)."""


def is_configured():
    """True when the server has what it needs to connect Gmail accounts."""
    return bool(CLIENT_ID and CLIENT_SECRET and REDIRECT_URI)


def _ensure_table():
    """Create the gmail_accounts table if it is missing (databases made before it existed)."""
    global _TABLE_READY
    if _TABLE_READY:
        return
    with db_cursor(commit=True) as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gmail_accounts (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                email TEXT NOT NULL,
                refresh_token TEXT NOT NULL,
                connected_at TIMESTAMP DEFAULT NOW()
            )""")
    _TABLE_READY = True


# ── Connecting ───────────────────────────────────────────────────────────────

def connect_url(user_id):
    """The Google page where the user allows DocQuery to read their Gmail.
    The state value says which DocQuery user started this; it is signed and
    expires in 10 minutes, so nobody can attach their Gmail to another account."""
    if not is_configured():
        raise GmailError("Gmail is not set up on the server (GOOGLE_CLIENT_SECRET is missing in backend/.env).")
    state = jwt.encode({"user_id": user_id, "purpose": "gmail-connect",
                        "exp": datetime.now(timezone.utc) + timedelta(minutes=10)},
                       SECRET_KEY, algorithm=ALGORITHM)
    return AUTH_URL + "?" + urlencode({
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",          # a refresh token, so it keeps working after an hour
        "prompt": "consent",               # always show the permission screen, so a refresh token is always sent
        "include_granted_scopes": "true",
        "state": state,
    })


def _user_from_state(state):
    try:
        data = jwt.decode(state or "", SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise GmailError("This Gmail connection link has expired. Please click Connect Gmail again.")
    if data.get("purpose") != "gmail-connect" or "user_id" not in data:
        raise GmailError("This Gmail connection link is not valid. Please click Connect Gmail again.")
    return data["user_id"]


def finish_connect(code, state):
    """Google sent the user back with a one-time code: swap it for tokens and store them.
    Returns (user_id, the Gmail address)."""
    user_id = _user_from_state(state)
    try:
        response = requests.post(TOKEN_URL, data={
            "code": code, "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI, "grant_type": "authorization_code",
        }, timeout=20)
    except requests.RequestException as e:
        raise GmailError(f"Could not reach Google ({type(e).__name__}). Please try again.")
    tokens = response.json() if response.content else {}
    if response.status_code != 200:
        print(f"GMAIL: token exchange failed ({response.status_code}): {tokens}")
        raise GmailError("Google did not accept the connection. Please try again.")

    if GMAIL_READONLY not in (tokens.get("scope") or "").split():
        # the user unticked the Gmail permission on Google's screen
        raise GmailError("Gmail access was not allowed. Please connect again and tick the Gmail permission.")
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise GmailError("Google did not send a long-term key. Please connect again.")
    try:
        info = google_id_token.verify_oauth2_token(tokens["id_token"], google_requests.Request(), CLIENT_ID)
        email = info["email"]
    except Exception as e:
        print(f"GMAIL: could not read the account email ({type(e).__name__}): {e}")
        raise GmailError("Could not confirm which Google account was connected. Please try again.")

    _ensure_table()
    with db_cursor(commit=True) as cur:
        cur.execute("""
            INSERT INTO gmail_accounts (user_id, email, refresh_token, connected_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (user_id) DO UPDATE
               SET email = EXCLUDED.email, refresh_token = EXCLUDED.refresh_token, connected_at = NOW()""",
            (user_id, email, _fernet.encrypt(refresh_token.encode()).decode()))
    with _tokens_lock:
        _access_tokens[user_id] = (tokens["access_token"], time.time() + int(tokens.get("expires_in", 3600)) - 60)
    return user_id, email


def status(user_id):
    """{"connected": True/False, "email": the Gmail address or None, "available": server set up}"""
    if not is_configured():
        return {"connected": False, "email": None, "available": False}
    _ensure_table()
    with db_cursor() as cur:
        cur.execute("SELECT email FROM gmail_accounts WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
    return {"connected": bool(row), "email": row[0] if row else None, "available": True}


def _forget(user_id):
    with db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM gmail_accounts WHERE user_id = %s", (user_id,))
    with _tokens_lock:
        _access_tokens.pop(user_id, None)


def disconnect(user_id):
    """Remove the connection, and tell Google to cancel DocQuery's access too."""
    _ensure_table()
    refresh_token = _stored_refresh_token(user_id, missing_ok=True)
    if refresh_token:
        try:
            requests.post(REVOKE_URL, data={"token": refresh_token}, timeout=10)
        except requests.RequestException as e:  # still removed here; the user can also remove it in their Google account
            print(f"GMAIL: could not revoke the token at Google ({type(e).__name__}): {e}")
    _forget(user_id)


# ── Calling the Gmail API ────────────────────────────────────────────────────

def _stored_refresh_token(user_id, missing_ok=False):
    _ensure_table()
    with db_cursor() as cur:
        cur.execute("SELECT refresh_token FROM gmail_accounts WHERE user_id = %s", (user_id,))
        row = cur.fetchone()
    if not row:
        if missing_ok:
            return None
        raise GmailNotConnected("Gmail is not connected. Click \"Connect Gmail\" in DocQuery first.")
    try:
        return _fernet.decrypt(row[0].encode()).decode()
    except InvalidToken:
        # the JWT secret was changed, so the stored key can't be read any more
        _forget(user_id)
        raise GmailNotConnected("The Gmail connection has to be made again. Click \"Connect Gmail\" in DocQuery.")


def _access_token(user_id, fresh=False):
    """A short-lived (1 hour) access token, made from the stored refresh token."""
    with _tokens_lock:
        cached = _access_tokens.get(user_id)
    if cached and not fresh and cached[1] > time.time():
        return cached[0]
    refresh_token = _stored_refresh_token(user_id)
    try:
        response = requests.post(TOKEN_URL, data={
            "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
            "refresh_token": refresh_token, "grant_type": "refresh_token",
        }, timeout=20)
    except requests.RequestException as e:
        raise GmailError(f"Could not reach Google ({type(e).__name__}).")
    data = response.json() if response.content else {}
    if response.status_code != 200:
        if data.get("error") == "invalid_grant":
            # the user removed DocQuery's access in their Google account (or the test-mode key expired)
            _forget(user_id)
            raise GmailNotConnected("Google no longer accepts the Gmail connection. Click \"Connect Gmail\" to connect again.")
        raise GmailError(f"Google refused to give Gmail access ({response.status_code}).")
    token = data["access_token"]
    with _tokens_lock:
        _access_tokens[user_id] = (token, time.time() + int(data.get("expires_in", 3600)) - 60)
    return token


def api_get(user_id, path, params=None):
    """GET a Gmail API path (for example "/messages") for this user; returns the JSON."""
    for attempt in range(2):
        token = _access_token(user_id, fresh=attempt > 0)
        try:
            response = requests.get(GMAIL_API + path, params=params,
                                    headers={"Authorization": f"Bearer {token}"}, timeout=30)
        except requests.RequestException as e:
            raise GmailError(f"Could not reach Gmail ({type(e).__name__}).")
        if response.status_code == 401 and attempt == 0:
            continue  # the access token ran out early: get a new one and try once more
        if response.status_code == 404:
            raise GmailError("That email or attachment was not found in Gmail.")
        if response.status_code == 403:
            raise GmailError("Gmail refused the request. Check that the Gmail API is enabled in Google Cloud.")
        if response.status_code != 200:
            raise GmailError(f"Gmail returned an error ({response.status_code}).")
        return response.json()
    raise GmailNotConnected("Gmail did not accept the connection. Click \"Connect Gmail\" to connect again.")
