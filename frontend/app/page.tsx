"use client";

import { useState, useRef, useEffect } from "react";
import AuthModal from "./components/AuthModal";
import AnswerTrace, { type Trace } from "./components/AnswerTrace";
import { API_URL } from "./config";

function isTokenExpired(token: string): boolean {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    return payload.exp * 1000 < Date.now();
  } catch {
    return true;
  }
}
interface StructuredTable {
  title: string;
  columns: string[];
  rows: string[][];
}

interface StructuredResult {
  document: string;
  pairs: { key: string; value: string }[];
  tables: StructuredTable[];
  note: string;
}

interface Structured {
  results: StructuredResult[];
}

interface Message {
  role: "user" | "assistant";
  content: string;
  structured?: Structured | null;
  sources?: string[];
  sourceRefs?: { file: string; pages: number[] }[];
  trace?: Trace | null; // the steps taken to find this answer
  action?: string; // how the backend handled it (for example EMAIL), sent back with follow-ups
  emails?: EmailCard[]; // emails an answer is about, shown as cards under it
  attachments?: string[]; // PDFs attached to this question (shown as file cards)
  reading?: string; // set while PDFs are still being read: this message is replaced when they are ready
  uploads?: UploadItem[]; // status cards for PDFs added without a question
}

interface EmailCard {
  from: string;
  date: string; // already readable, in the user's time ("Today, 4:25 PM")
  subject: string;
  preview: string;
  attachments: { filename: string; in_documents: boolean }[];
}

interface UploadItem {
  name: string;
  status: "reading" | "ready" | "failed";
  note?: string; // for example "Already uploaded", or why reading failed
}

const MAX_ATTACHMENTS = 10;

// Numbers (with commas, decimals, %) line up on the right in tables.
const NUMERIC_CELL = /^[-+]?[\d,]*\.?\d+%?$/;

function tableToDelimited(table: StructuredTable, delimiter: string): string {
  const escape = (cell: string) =>
    delimiter === "," && /[",\n]/.test(cell)
      ? `"${cell.replace(/"/g, '""')}"`
      : cell;
  return [table.columns, ...table.rows]
    .map((row) => row.map(escape).join(delimiter))
    .join("\n");
}


// A column counts as numeric when every filled cell in it is a number.
function numericColumns(table: StructuredTable): boolean[] {
  return table.columns.map((_, c) => {
    const cells = table.rows.map((r) => (r[c] ?? "").trim()).filter(Boolean);
    return cells.length > 0 && cells.every((cell) => NUMERIC_CELL.test(cell));
  });
}

// Plain-text answers sometimes contain a markdown table ("| a | b |" lines).
// Turn those into real tables so they never show up as one long line of pipes.
type Block =
  | { kind: "text"; text: string }
  | { kind: "table"; table: StructuredTable };

function splitCells(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((c) => c.replace(/\*\*/g, "").trim()); // the model often bolds cells with **
}

function parseBlocks(text: string): Block[] {
  const lines = text.split("\n");
  const isRow = (l: string) => /^\s*\|.*\|\s*$/.test(l);
  const isSeparator = (l: string) => isRow(l) && /^[\s|:-]+$/.test(l) && l.includes("-");
  const blocks: Block[] = [];
  let buffer: string[] = [];

  const flush = () => {
    if (buffer.length) blocks.push({ kind: "text", text: buffer.join("\n") });
    buffer = [];
  };

  let i = 0;
  while (i < lines.length) {
    if (isRow(lines[i]) && i + 1 < lines.length && isSeparator(lines[i + 1])) {
      const columns = splitCells(lines[i]);
      const rows: string[][] = [];
      let j = i + 2;
      while (j < lines.length && isRow(lines[j]) && !isSeparator(lines[j])) {
        const cells = splitCells(lines[j]);
        rows.push([...cells, ...Array(columns.length).fill("")].slice(0, columns.length));
        j++;
      }
      flush();
      blocks.push({ kind: "table", table: { title: "", columns, rows } });
      i = j;
    } else {
      buffer.push(lines[i]);
      i++;
    }
  }
  flush();
  return blocks;
}

function TableView({ table }: { table: StructuredTable }) {
  const numericCols = numericColumns(table);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      // Tab-separated, so it pastes into Excel / Google Sheets as real cells.
      await navigator.clipboard.writeText(tableToDelimited(table, "\t"));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard not available */
    }
  };

  const handleDownload = () => {
    // "﻿" (BOM) tells Excel the file is UTF-8, so symbols like "–" show correctly
    const blob = new Blob(["﻿" + tableToDelimited(table, ",")], {
      type: "text/csv;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${(table.title || "table").replace(/[^\w\- ]+/g, "").trim() || "table"}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mt-3 first:mt-0">
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <p className="text-[13px] font-semibold text-[#1C1C1E]">
          {table.title}
          <span className={`${table.title ? "ml-2" : ""} text-[11px] font-normal text-[#9A968C]`}>
            {table.rows.length} {table.rows.length === 1 ? "row" : "rows"}
          </span>
        </p>
        <div className="flex flex-shrink-0 items-center gap-1.5">
          <button
            onClick={handleCopy}
            className="rounded-md border border-[#E6E0D2] bg-white px-2 py-0.5 text-[11px] font-medium text-[#6B6B6B] transition hover:border-[#B8CFB3] hover:bg-[#EEF4EC] hover:text-[#5E7C5C]"
          >
            {copied ? "Copied" : "Copy"}
          </button>
          <button
            onClick={handleDownload}
            className="rounded-md border border-[#E6E0D2] bg-white px-2 py-0.5 text-[11px] font-medium text-[#6B6B6B] transition hover:border-[#B8CFB3] hover:bg-[#EEF4EC] hover:text-[#5E7C5C]"
          >
            CSV
          </button>
        </div>
      </div>

      <div className="inline-block max-h-[420px] max-w-full overflow-auto rounded-xl border border-[#E0DBD3] align-top">
        <table className="w-auto table-auto border-collapse text-[12.5px]">
          <thead className="sticky top-0 bg-[#EEF4EC] text-[#5E7C5C]">
            <tr>
              {table.columns.map((col, c) => (
                <th
                  key={c}
                  className={`border-b border-[#B8CFB3] px-3 py-2 align-bottom text-[11.5px] font-semibold leading-tight ${
                    numericCols[c] ? "text-right" : "text-left"
                  }`}
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.length === 0 ? (
              <tr>
                <td
                  colSpan={Math.max(table.columns.length, 1)}
                  className="px-3 py-3 text-center text-[#9A968C]"
                >
                  No rows
                </td>
              </tr>
            ) : (
              table.rows.map((row, r) => (
                <tr key={r} className={r % 2 === 1 ? "bg-[#FAF8F5]" : "bg-white"}>
                  {row.map((cell, c) => (
                    <td
                      key={c}
                      className={`border-b border-[#EFEBE4] px-3 py-1.5 align-top ${
                        numericCols[c]
                          ? "whitespace-nowrap text-right tabular-nums"
                          : cell.length <= 14
                            ? "whitespace-nowrap"
                            : ""
                      }`}
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PairsView({ pairs }: { pairs: { key: string; value: string }[] }) {
  return (
    <div className="overflow-hidden rounded-xl border border-[#E0DBD3]">
      <table className="w-full border-collapse text-[13px]">
        <tbody>
          {pairs.map((p, i) => (
            <tr key={i} className={i % 2 === 1 ? "bg-[#FAF8F5]" : "bg-white"}>
              <th className="w-[38%] border-b border-[#EFEBE4] bg-[#F6FAF4] px-3 py-1.5 text-left align-top text-[12px] font-semibold text-[#5E7C5C]">
                {p.key}
              </th>
              <td className="border-b border-[#EFEBE4] px-3 py-1.5 align-top">
                {p.value}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StructuredView({ structured }: { structured: Structured }) {
  const showDocNames = structured.results.length > 1;
  return (
    <div className="space-y-4">
      {structured.results.map((result, i) => (
        <div key={i} className="space-y-3">
          {showDocNames && (
            <div className="flex items-center gap-1.5 text-[12px] font-medium text-[#5E7C5C]">
              <FileIcon /> {result.document}
            </div>
          )}
          {result.pairs.length > 0 && <PairsView pairs={result.pairs} />}
          {result.tables.map((table, t) => (
            <TableView key={t} table={table} />
          ))}
          {result.note && (
            <p className="text-[12.5px] italic text-[#9A968C]">{result.note}</p>
          )}
        </div>
      ))}
    </div>
  );
}


function UploadIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function FileIcon() {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}

function MailIcon() {
  return (
    <svg
      className="h-3.5 w-3.5"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.8}
      aria-hidden="true"
    >
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.5 6.5l8.5 6 8.5-6" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

function BotIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M13 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h4" />
      <path d="M13 3v4a1 1 0 0 0 1 1h4" />
      <path d="M13 3l5 5v3" />
      <line x1="8.5" y1="9.5" x2="11" y2="9.5" />
      <line x1="8.5" y1="13" x2="12" y2="13" />
      <circle cx="16" cy="16.5" r="3" />
      <line x1="18.2" y1="18.7" x2="20.5" y2="21" />
    </svg>
  );
}

// Emails an answer is about, drawn by the app (not written by the model), so they always
// look the same: sender and date, subject, a preview line and the attached files.
function EmailCards({
  emails,
  onOpen,
}: {
  emails: EmailCard[];
  onOpen: (name: string) => void;
}) {
  return (
    <ul className="mt-3 space-y-2">
      {emails.map((e, i) => (
        <li key={i} className="rounded-xl border border-[#E9E4DA] bg-[#FBFAF7] px-3.5 py-2.5">
          <div className="flex items-baseline justify-between gap-3">
            <span className="min-w-0 truncate text-[13px] font-semibold text-[#1C1C1E]">{e.from}</span>
            <span className="flex-shrink-0 text-[11.5px] text-[#9A968C]">{e.date}</span>
          </div>
          <p className="mt-0.5 truncate text-[13px] text-[#3A3A3C]">
            {e.subject && e.subject !== "(no subject)" ? (
              e.subject
            ) : (
              <span className="italic text-[#9A968C]">No subject</span>
            )}
          </p>
          {e.preview && (
            <p className="mt-0.5 line-clamp-2 text-[12px] leading-snug text-[#8A867D]">{e.preview}</p>
          )}
          {e.attachments.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {e.attachments.map((a) =>
                a.in_documents ? (
                  <button
                    key={a.filename}
                    onClick={() => onOpen(a.filename)}
                    title={`In your documents · open ${a.filename}`}
                    className="flex max-w-full items-center gap-1.5 rounded-lg border border-[#B8CFB3] bg-[#EEF4EC] px-2 py-1 text-[11.5px] text-[#3F5A3D] transition hover:border-[#5E7C5C]"
                  >
                    <span className="rounded bg-white px-1 text-[9px] font-bold text-red-600">PDF</span>
                    <span className="truncate">{a.filename}</span>
                    <span className="flex-shrink-0 text-[#6F8F6C]" aria-label="in your documents">
                      <span className="sm:hidden">✓</span>
                      <span className="hidden sm:inline">· in your documents</span>
                    </span>
                  </button>
                ) : (
                  <span
                    key={a.filename}
                    className="flex max-w-full items-center gap-1.5 rounded-lg border border-[#E6E0D2] bg-white px-2 py-1 text-[11.5px] text-[#6B6B6B]"
                  >
                    <span
                      className={`rounded px-1 text-[9px] font-bold ${
                        a.filename.toLowerCase().endsWith(".pdf") ? "bg-red-50 text-red-600" : "bg-[#F4F1EA] text-[#8A867D]"
                      }`}
                    >
                      {(a.filename.split(".").pop() ?? "FILE").toUpperCase().slice(0, 4)}
                    </span>
                    <span className="truncate">{a.filename}</span>
                  </span>
                ),
              )}
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploadedDocs, setUploadedDocs] = useState<string[]>([]);
  const watchedJobs = useRef<Set<string>>(new Set()); // upload jobs already being followed
  const tokenRef = useRef<string | null>(null); // always the current login token
  const recentUploads = useRef<string[]>([]); // the files added last, so "these two documents" can be understood
  const [token, setToken] = useState<string | null>(null);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "signup">("login");
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showDocs, setShowDocs] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);
  const [attachments, setAttachments] = useState<File[]>([]);
  const [busyLabel, setBusyLabel] = useState<string | null>(null);
  // Gmail: connected or not (null until known), and a short message after connecting
  // (kept with the login it belongs to, so a different login never sees the last one's Gmail)
  const [gmailState, setGmailState] = useState<{
    token: string;
    available: boolean;
    connected: boolean;
    email: string | null;
  } | null>(null);
  const gmail = gmailState && gmailState.token === token ? gmailState : null;
  const [showGmail, setShowGmail] = useState(false);
  const [gmailBusy, setGmailBusy] = useState(false);
  const [gmailNotice, setGmailNotice] = useState<{ ok: boolean; text: string } | null>(null);

  // Pop-ups (My documents, account menu) close when you click anywhere else or press Esc.
  useEffect(() => {
    const closePopups = () => {
      setShowDocs(false);
      setShowUserMenu(false);
      setShowGmail(false);
      setConfirmDelete(null);
    };
    const onClick = (e: MouseEvent) => {
      if (!(e.target as Element | null)?.closest?.("[data-popup]")) closePopups();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closePopups();
    };
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (!token) return;
    fetch(`${API_URL}/documents`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => setUploadedDocs(data.documents ?? []))
      .catch(() => {});
  }, [token]);

  useEffect(() => {
    tokenRef.current = token;
  }, [token]);

  // Is Gmail connected? Asked again whenever the login changes.
  useEffect(() => {
    if (!token) return;
    fetch(`${API_URL}/gmail/status`, { headers: { Authorization: `Bearer ${token}` } })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setGmailState(data ? { ...data, token } : null))
      .catch(() => setGmailState(null));
  }, [token]);

  // Coming back from Google's permission page: ?gmail=connected or ?gmail_error=...
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const connected = params.get("gmail") === "connected";
    const error = params.get("gmail_error");
    if (!connected && !error) return;
    window.history.replaceState(null, "", window.location.pathname); // so a refresh doesn't show it again
    // (not cancelled on cleanup: in development React runs this twice, and the second run
    // finds the address already cleaned, so the first run's message must still appear)
    setTimeout(() =>
      setGmailNotice(
        connected
          ? { ok: true, text: "Gmail connected. Ask me about your emails, or to import a PDF attachment." }
          : { ok: false, text: error ?? "Gmail was not connected." },
      ),
    );
  }, []);

  useEffect(() => {
    if (!gmailNotice) return;
    const timer = setTimeout(() => setGmailNotice(null), 7000);
    return () => clearTimeout(timer);
  }, [gmailNotice]);

  // "Connect Gmail": the backend gives the Google page to go to; Google sends the
  // browser back to the app afterwards.
  const connectGmail = async () => {
    if (!token) return;
    setGmailBusy(true);
    try {
      const res = await fetch(`${API_URL}/gmail/connect`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (!res.ok || !data.url) throw new Error(data.detail ?? "Gmail can't be connected right now.");
      window.location.href = data.url;
    } catch (e) {
      setGmailNotice({ ok: false, text: e instanceof Error ? e.message : "Gmail can't be connected right now." });
      setGmailBusy(false);
    }
  };

  const disconnectGmail = async () => {
    if (!token) return;
    setGmailBusy(true);
    try {
      const res = await fetch(`${API_URL}/gmail`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        setGmailState((prev) => (prev ? { ...prev, connected: false, email: null } : prev));
        setShowGmail(false);
      }
    } finally {
      setGmailBusy(false);
    }
  };

  // ── Chat history: saved on the server, one history per user ──
  // `chatReadyFor` is the user whose saved chat has been loaded. Nothing is
  // saved before that, so a failed load can never overwrite the real history.
  const localChatKey = userEmail ? `chatHistory:${userEmail}` : null;
  const [chatReadyFor, setChatReadyFor] = useState<string | null>(null);
  const [authChecked, setAuthChecked] = useState(false); // the saved login has been looked up
  const [chatChecked, setChatChecked] = useState<string | null>(null); // the user whose saved chat has been tried

  // When a user is logged in (or logs in again), bring back their saved chat.
  useEffect(() => {
    if (!token || !userEmail) {
      setChatReadyFor(null);
      setChatChecked(null);
      return;
    }
    let cancelled = false;
    const oldLocalKey = `chatHistory:${userEmail}`;
    const readLocal = (): Message[] => {
      try {
        const raw = localStorage.getItem(oldLocalKey);
        return raw ? JSON.parse(raw) : [];
      } catch {
        return [];
      }
    };

    fetch(`${API_URL}/chat`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((data) => {
        if (cancelled) return;
        let saved: Message[] = Array.isArray(data.messages) ? data.messages : [];
        // First time on the server: bring over the chat kept in this browser before.
        if (saved.length === 0) saved = readLocal();
        setMessages(saved);
        try {
          localStorage.removeItem(oldLocalKey);
        } catch {}
        setChatReadyFor(userEmail);
        setChatChecked(userEmail);
      })
      .catch(() => {
        // Server not reachable: show the browser copy, but do not save anything.
        if (!cancelled) {
          setMessages(readLocal());
          setChatChecked(userEmail);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [token, userEmail]);

  // Save the chat shortly after it changes (only the last 100 messages are kept).
  useEffect(() => {
    if (!token || !userEmail || chatReadyFor !== userEmail) return;
    const timer = setTimeout(() => {
      fetch(`${API_URL}/chat`, {
        method: "PUT",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        // a "Reading…" message is temporary, it is never saved
        body: JSON.stringify({ messages: messages.filter((m) => !m.reading).slice(-100) }),
      }).catch(() => {
        /* saving failed - the chat still works, it just won't be saved this time */
      });
    }, 800);
    return () => clearTimeout(timer);
  }, [messages, token, userEmail, chatReadyFor]);

  const handleClearChat = () => {
    setMessages([]);
    if (token) {
      fetch(`${API_URL}/chat`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      }).catch(() => {});
    }
    if (localChatKey) {
      try {
        localStorage.removeItem(localChatKey);
      } catch {}
    }
  };

  useEffect(() => {
    const savedToken = localStorage.getItem("token");
    const savedEmail = localStorage.getItem("userEmail");
    if (savedToken && savedEmail) {
      if (isTokenExpired(savedToken)) {
        localStorage.removeItem("token");
        localStorage.removeItem("userEmail");
      } else {
        setToken(savedToken);
        setUserEmail(savedEmail);
      }
    }
    setAuthChecked(true);
  }, []);


   const handleSessionExpired = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("userEmail");
    setToken(null);
    setUserEmail(null);
    setUploadedDocs([]);
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "Your session expired — please log in again." },
    ]);
  };

  const handleDeleteDocument = async (name: string) => {
    if (!token || isTokenExpired(token)) {
      setConfirmDelete(null);
      setShowDocs(false);
      handleSessionExpired();
      return;
    }
    setDeletingDoc(name);
    try {
      const res = await fetch(
        `${API_URL}/documents?name=${encodeURIComponent(name)}`,
        { method: "DELETE", headers: { Authorization: `Bearer ${token}` } },
      );
      if (!res.ok) throw new Error("delete failed");
      // The file vanishing from the list (and the count dropping) is the feedback.
      setUploadedDocs((prev) => prev.filter((d) => d !== name));
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `I couldn't delete "${name}". Please try again.` },
      ]);
    } finally {
      setDeletingDoc(null);
      setConfirmDelete(null);
    }
  };

  // Open a stored PDF in a new tab (at a given page, if we know it).
  // The file is fetched with the login token and shown from memory, because a
  // plain link could not carry the token.
  const openDocument = async (name: string, page?: number) => {
    if (!token || isTokenExpired(token)) {
      handleSessionExpired();
      return;
    }
    // Open the tab right away: browsers block tabs opened after a delay.
    const win = window.open("", "_blank");
    if (win) win.document.title = "Opening…";
    const fail = (text: string) => {
      win?.close();
      setMessages((prev) => [...prev, { role: "assistant", content: text }]);
    };
    try {
      const res = await fetch(
        `${API_URL}/documents/file?name=${encodeURIComponent(name)}`,
        { headers: { Authorization: `Bearer ${token}` } },
      );
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        fail(data.detail ?? `I couldn't open "${name}".`);
        return;
      }
      const url = URL.createObjectURL(await res.blob());
      const target = page ? `${url}#page=${page}` : url;
      if (win) win.location.href = target;
      else window.open(target, "_blank");
      setTimeout(() => URL.revokeObjectURL(url), 10 * 60 * 1000);
    } catch {
      fail(`I couldn't open "${name}". Please try again.`);
    }
  };

  // Ask the backend a question. `scope` limits it to some documents.
  // `past` is the chat as it was before this question (for follow-ups).
  const askQuestion = async (question: string, past: Message[], scope?: string[]) => {
    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;

      // Build recent history from the last few exchanges (question/answer pairs)
      // so the backend can resolve follow-ups like "that" or "the second one".
      // The sources go along too, so "make it shorter" can keep the same sources.
      const recentHistory: {
        question: string;
        answer: string;
        sources: string[];
        source_refs: { file: string; pages: number[] }[];
        attachments: string[];
        action?: string;
      }[] = [];
      for (let i = 0; i < past.length - 1; i++) {
        if (past[i].role === "user" && past[i + 1]?.role === "assistant" && !past[i + 1].uploads) {
          recentHistory.push({
            question: past[i].content,
            answer: past[i + 1].content,
            sources: past[i + 1].sources ?? [],
            source_refs: past[i + 1].sourceRefs ?? [],
            attachments: past[i].attachments ?? [], // files attached to this question
            action: past[i + 1].action, // "EMAIL" lets "import the second one" stay about emails
          });
        }
      }
      const lastFewTurns = recentHistory.slice(-6);

      // Everything the user typed in this chat, so small details they stated
      // earlier ("my name is ...") can be remembered.
      const chatUserMessages = past
        .filter((m) => m.role === "user")
        .map((m) => m.content.slice(0, 200))
        .slice(-60);

      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          question,
          history: lastFewTurns,
          chat_user_messages: chatUserMessages,
          ...(scope && scope.length ? { source_files: scope } : {}),
          recent_uploads: recentUploads.current,
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone, // email times in the user's own time
        }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          structured: data.structured?.results?.length ? data.structured : null,
          sources: Array.isArray(data.sources) ? data.sources : [],
          sourceRefs: Array.isArray(data.source_refs) ? data.source_refs : [],
          trace: data.trace?.steps?.length ? data.trace : null,
          action: data.action_taken,
          emails: Array.isArray(data.emails) && data.emails.length ? data.emails : undefined,
        },
      ]);
      // PDFs the agent imported from Gmail: show them being read, like an upload
      const imported = Array.isArray(data.imported) ? data.imported : [];
      if (imported.length && token) {
        void watchInBackground(
          imported.map((j: { job_id: string; filename: string }) => ({ id: j.job_id, name: j.filename })),
          token,
        );
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I couldn't reach the server. Please try again.",
        },
      ]);
    }
  };

  // Send one PDF to the server. The server answers at once and reads the file in the
  // background, so this returns a job to follow (or says the file was already there).
  const uploadOne = async (
    file: File,
  ): Promise<{ name?: string; already?: boolean; jobId?: string; error?: string }> => {
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      const data = await res.json();
      if (!res.ok || !data.filename) {
        return { error: data.detail ?? `I couldn't process "${file.name}".` };
      }
      if (data.status === "reading") {
        return { name: data.filename, jobId: data.job_id };
      }
      setUploadedDocs((prev) => (prev.includes(data.filename) ? prev : [data.filename, ...prev]));
      return { name: data.filename, already: data.chunks_stored === 0 };
    } catch {
      return { error: `I couldn't process "${file.name}".` };
    }
  };

  // One file is finished reading: add it to the document list.
  const finishDoc = (name: string, ok: boolean) => {
    if (ok) setUploadedDocs((prev) => (prev.includes(name) ? prev : [name, ...prev])); // newest first
  };

  // Follow upload jobs until each one is done or has failed. onFinished is called
  // once per file. A lost connection is retried for a while, then counts as a failure.
  const waitForJobs = async (
    jobs: { id: string; name: string }[],
    authToken: string,
    onFinished?: (job: { id: string; name: string }, ok: boolean, error?: string) => void,
  ) => {
    const results: Record<string, { ok: boolean; error?: string }> = {};
    let misses = 0;
    while (Object.keys(results).length < jobs.length) {
      if (tokenRef.current !== authToken) return results; // logged out meanwhile
      await new Promise((r) => setTimeout(r, 1500));
      const pending = jobs.filter((j) => !(j.id in results));
      let status: Record<string, { status: string; error?: string }> | null = null;
      try {
        const res = await fetch(
          `${API_URL}/upload/status?ids=${pending.map((j) => j.id).join(",")}`,
          { headers: { Authorization: `Bearer ${authToken}` } },
        );
        status = res.ok ? ((await res.json()).jobs ?? null) : null;
      } catch {
        status = null;
      }
      if (!status) {
        misses += 1;
        if (misses < 6) continue;
        for (const j of pending) {
          results[j.id] = { ok: false, error: "I lost the connection to the server." };
          finishDoc(j.name, false);
          onFinished?.(j, false, results[j.id].error);
        }
        break;
      }
      misses = 0;
      for (const j of pending) {
        const info = status[j.id];
        if (!info || info.status === "reading") continue;
        const ok = info.status === "done";
        results[j.id] = { ok, error: info.error ?? undefined };
        finishDoc(j.name, ok);
        onFinished?.(j, ok, info.error ?? undefined);
      }
    }
    return results;
  };

  // Files sent without a question: status cards appear in the chat (Reading… turns into
  // Ready when the file is done). The chat stays free to use meanwhile.
  const watchInBackground = async (
    jobs: { id: string; name: string }[],
    authToken: string,
    alsoThere: string[] = [],
  ) => {
    jobs.forEach((j) => watchedJobs.current.add(j.id));
    const key = jobs.map((j) => j.id).join(",");
    const known: UploadItem[] = alsoThere.map((name) => ({ name, status: "ready", note: "Already uploaded" }));
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: "Reading documents",
        reading: key,
        uploads: [...known, ...jobs.map((j) => ({ name: j.name, status: "reading" as const }))],
      },
    ]);
    const results = await waitForJobs(jobs, authToken);
    if (tokenRef.current !== authToken) return;
    const ready = jobs.filter((j) => results[j.id]?.ok).map((j) => j.name);
    if (ready.length + alsoThere.length > 0) recentUploads.current = [...alsoThere, ...ready];
    const finished: UploadItem[] = jobs.map((j) =>
      results[j.id]?.ok
        ? { name: j.name, status: "ready" as const }
        : { name: j.name, status: "failed" as const, note: results[j.id]?.error ?? "This PDF could not be read." },
    );
    const done: Message = {
      role: "assistant",
      content: "Documents finished reading",
      uploads: [...known, ...finished],
    };
    setMessages((prev) =>
      prev.some((m) => m.reading === key)
        ? prev.map((m) => (m.reading === key ? done : m))
        : [...prev, done],
    );
  };

  // Files that were still being read when the page was opened (for example after a refresh).
  useEffect(() => {
    if (!token || !userEmail || chatReadyFor !== userEmail) return; // wait until the saved chat is loaded
    fetch(`${API_URL}/upload/status`, { headers: { Authorization: `Bearer ${token}` } })
      .then((res) => res.json())
      .then((data) => {
        const active = Object.values(data.jobs ?? {}) as { job_id: string; filename: string }[];
        const fresh = active.filter((j) => !watchedJobs.current.has(j.job_id));
        if (fresh.length === 0) return;
        void watchInBackground(fresh.map((j) => ({ id: j.job_id, name: j.filename })), token);
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, userEmail, chatReadyFor]);

  const handleSend = async () => {
    if (loading) return;
    const question = input.trim();
    const files = attachments;
    if (!question && files.length === 0) return;

    if (files.length > 0 && (!token || isTokenExpired(token))) {
      handleSessionExpired();
      return;
    }

    const past = messages;
    const questionIndex = past.length;
    if (question) {
      setMessages((prev) => [
        ...prev,
        {
          role: "user",
          content: question,
          ...(files.length ? { attachments: files.map((f) => f.name) } : {}),
        },
      ]);
    }
    setInput("");
    setAttachments([]);
    setLoading(true);

    try {
      let scope: string[] | undefined;
      if (files.length > 0) {
        // Send the attached files first (they join the knowledge base once they are read).
        const stored: string[] = [];
        const already: string[] = [];
        const jobs: { id: string; name: string }[] = [];
        for (let i = 0; i < files.length; i++) {
          setBusyLabel(
            files.length > 1
              ? `Sending document ${i + 1} of ${files.length}…`
              : "Sending your document…",
          );
          const result = await uploadOne(files[i]);
          if (result.name) {
            stored.push(result.name);
            if (result.jobId) jobs.push({ id: result.jobId, name: result.name });
            else if (result.already) already.push(result.name);
          } else {
            setMessages((prev) => [
              ...prev,
              { role: "assistant", content: result.error ?? "I couldn't process that file." },
            ]);
          }
        }

        if (!question) {
          // Only files, no question: the server reads them in the background, the chat stays free.
          setBusyLabel(null);
          if (jobs.length > 0) {
            // status cards in the chat show the progress
            void watchInBackground(jobs, token as string, already);
          } else if (already.length) {
            recentUploads.current = [...already];
            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content: "Already uploaded",
                uploads: already.map((name) => ({ name, status: "ready" as const, note: "Already uploaded" })),
              },
            ]);
          }
          return;
        }

        // A question about the files: wait until they are read, showing the progress.
        if (jobs.length > 0) {
          const total = files.length;
          const alreadyDone = stored.length - jobs.length;
          let finished = 0;
          const label = () =>
            total > 1
              ? `Reading document ${Math.min(alreadyDone + finished + 1, total)} of ${total}…`
              : "Reading your document…";
          setBusyLabel(label());
          jobs.forEach((j) => watchedJobs.current.add(j.id));
          const results = await waitForJobs(jobs, token as string, () => {
            finished += 1;
            setBusyLabel(label());
          });
          for (const j of jobs) {
            const r = results[j.id];
            if (!r || !r.ok) {
              const index = stored.indexOf(j.name);
              if (index >= 0) stored.splice(index, 1);
              setMessages((prev) => [
                ...prev,
                { role: "assistant", content: `I couldn't read "${j.name}". ${r?.error ?? ""}`.trim() },
              ]);
            }
          }
        }
        setBusyLabel(null);

        if (stored.length === 0) return; // nothing could be read, nothing to ask about
        scope = [...new Set(stored)];
        recentUploads.current = scope;
        // show the names the documents are really stored under
        setMessages((prev) =>
          prev.map((m, i) => (i === questionIndex ? { ...m, attachments: scope } : m)),
        );
      }
      await askQuestion(question, past, scope);
    } finally {
      setBusyLabel(null);
      setLoading(false);
    }
  };

  // Put PDFs next to the question box (they are uploaded when the message is sent).
  // Uploading needs an account: if the user is not logged in, show the login popup instead.
  // Returns true when the user is logged in and may continue.
  const requireLogin = () => {
    if (token && !isTokenExpired(token)) return true;
    if (token) handleSessionExpired(); // the login has run out: say so
    setShowAuthModal(true);
    return false;
  };

  const attachFiles = (files: File[]) => {
    if (!requireLogin()) return;
    const pdfs = files.filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length < files.length) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Only PDF files are supported. Please attach a valid PDF." },
      ]);
    }
    setAttachments((prev) => [...prev, ...pdfs].slice(0, MAX_ATTACHMENTS));
    inputRef.current?.focus();
  };

  const handleAttach = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) attachFiles(Array.from(files));
    e.target.value = "";
  };

  const handleAuthSuccess = (newToken: string, email: string) => {
    localStorage.setItem("token", newToken);
    localStorage.setItem("userEmail", email);
    setToken(newToken);
    setUserEmail(email);
    setShowAuthModal(false);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("userEmail");
    setToken(null);
    setUserEmail(null);
    setUploadedDocs([]);
    setMessages([]); // the saved chat stays on the server for the next login
  };

  // "Drop your PDF here" box. Moving the file over different parts of the page fires
  // dragenter/dragleave many times (once per element), so they are counted: the box stays
  // until the file has really left the page. (Hiding it on every dragleave made it flicker,
  // and a drop in that moment was handed to the browser, which opened the PDF.)
  const dragDepth = useRef(0);

  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    if (!e.dataTransfer?.types?.includes("Files")) return;
    e.preventDefault();
    dragDepth.current += 1;
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setIsDragging(false);
  };

  // Safety net: a file dropped anywhere on the window must never make the browser open it.
  useEffect(() => {
    const stop = (e: DragEvent) => {
      if (e.dataTransfer?.types?.includes("Files")) e.preventDefault();
    };
    window.addEventListener("dragover", stop);
    window.addEventListener("drop", stop);
    return () => {
      window.removeEventListener("dragover", stop);
      window.removeEventListener("drop", stop);
    };
  }, []);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragDepth.current = 0;
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      attachFiles(Array.from(files));
    }
  };

  // Render **bold** markdown inline, keep line breaks, and show any
  // markdown table in the text as a real table.
  const renderBold = (text: string) =>
    text
      .split(/(\*\*[^*]+\*\*)/)
      .map((part, i) =>
        part.startsWith("**") && part.endsWith("**") ? (
          <strong key={i}>{part.slice(2, -2)}</strong>
        ) : (
          <span key={i}>{part}</span>
        ),
      );
  // Lines that start with "* ", "- " or "• " are shown as normal bullets
  // (a real • with a neat hanging indent) instead of the raw characters.
  const renderLines = (text: string) =>
    text.split("\n").map((line, i) => {
      const bullet = line.match(/^(\s*)[*\-•]\s+(.*)$/);
      if (bullet) {
        const level = Math.min(Math.floor(bullet[1].length / 2), 3);
        return (
          <div key={i} className="flex gap-2" style={{ paddingLeft: `${0.5 + level * 1.25}rem` }}>
            <span aria-hidden="true">•</span>
            <span className="whitespace-pre-wrap">{renderBold(bullet[2])}</span>
          </div>
        );
      }
      if (line.trim() === "") return <div key={i} className="h-2" />;
      return (
        <div key={i} className="whitespace-pre-wrap">
          {renderBold(line)}
        </div>
      );
    });
  const renderContent = (text: string) =>
    parseBlocks(text).map((block, i) =>
      block.kind === "table" ? (
        <div key={i} className="my-2">
          <TableView table={block.table} />
        </div>
      ) : (
        <div key={i}>{renderLines(block.text)}</div>
      ),
    );
  const userInitial = userEmail ? userEmail[0].toUpperCase() : "U";
  // While the saved login and chat are being brought back, show a quiet placeholder
  // instead of flashing the empty welcome screen.
  const chatLoading = !authChecked || (!!token && !!userEmail && chatChecked !== userEmail);
  const isEmpty = messages.length === 0 && !chatLoading;

  return (
    <div
      className="flex h-screen flex-col bg-[#FEFAF3]"
      onDragEnter={handleDragEnter}
      onDragOver={(e) => {
        if (e.dataTransfer?.types?.includes("Files")) e.preventDefault(); // allow dropping here
      }}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag-and-drop overlay */}
      {isDragging && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#5E7C5C]/10 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-3 rounded-2xl border-2 border-dashed border-[#5E7C5C] bg-white/80 px-12 py-10 text-[#5E7C5C]">
            <UploadIcon />
            <p className="text-base font-semibold">Drop your PDF here</p>
          </div>
        </div>
      )}

      {/* ── Top bar ───────────────────────────────── */}
      <header className="relative z-30 flex items-center justify-between gap-3 border-b border-[#E6E0D2] bg-[#FEFAF3]/95 px-4 py-3 backdrop-blur sm:gap-4 sm:px-6">
        {/* Brand */}
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-[#5E7C5C] text-white shadow-sm [&_svg]:h-[18px] [&_svg]:w-[18px]">
            <BotIcon />
          </div>
          <div className="min-w-0">
            <h1 className="text-[15px] font-semibold tracking-tight text-[#1C1C1E] leading-tight">
              DocQuery
            </h1>
          </div>
        </div>

        {/* Spacer keeps the upload and account buttons on the right */}
        <div className="flex-1" />

        {/* Gmail */}
        {token && gmail?.available && (
          <div className="relative flex-shrink-0" data-popup>
            {gmail.connected ? (
              <button
                onClick={() => {
                  setShowGmail((prev) => !prev);
                  setShowDocs(false);
                  setShowUserMenu(false);
                  setConfirmDelete(null);
                }}
                aria-expanded={showGmail}
                title={`Gmail connected: ${gmail.email}`}
                className="flex items-center gap-1.5 rounded-full border border-[#E6E0D2] bg-white px-3 py-1.5 text-xs font-medium text-[#6B6B6B] shadow-sm transition hover:border-[#B8CFB3] hover:text-[#5E7C5C]"
              >
                <MailIcon />
                <span className="hidden sm:inline">Gmail</span>
                <span className="h-1.5 w-1.5 rounded-full bg-[#5E7C5C]" aria-label="connected" />
              </button>
            ) : (
              <button
                onClick={connectGmail}
                disabled={gmailBusy}
                title="Let DocQuery read your Gmail (read-only)"
                className="flex items-center gap-1.5 rounded-full border border-dashed border-[#C9C2B2] bg-white px-3 py-1.5 text-xs font-medium text-[#6B6B6B] transition hover:border-[#5E7C5C] hover:text-[#5E7C5C] disabled:opacity-60"
              >
                <MailIcon />
                <span className="hidden sm:inline">{gmailBusy ? "Opening Google…" : "Connect Gmail"}</span>
              </button>
            )}
            {showGmail && gmail.connected && (
              <div className="absolute right-0 top-11 z-20 w-72 max-w-[calc(100vw-2rem)] rounded-2xl border border-[#E6E0D2] bg-white p-4 shadow-xl">
                <p className="text-[13px] font-semibold text-[#1C1C1E]">Gmail connected</p>
                <p className="mt-0.5 truncate text-[12.5px] text-[#6B6B6B]">{gmail.email}</p>
                <p className="mt-2.5 text-[12px] leading-relaxed text-[#9A968C]">
                  Read-only: I can find and read your emails and import PDF attachments, but never send or
                  delete anything. Try &ldquo;Any emails with a PDF this week?&rdquo;
                </p>
                <button
                  onClick={disconnectGmail}
                  disabled={gmailBusy}
                  className="mt-3 rounded-md border border-[#E6E0D2] px-2.5 py-1 text-[12px] text-[#6B6B6B] transition hover:border-red-200 hover:bg-red-50 hover:text-red-600 disabled:opacity-50"
                >
                  {gmailBusy ? "Disconnecting…" : "Disconnect Gmail"}
                </button>
              </div>
            )}
          </div>
        )}

        {/* My documents */}
        {token && (
          <div className="relative flex-shrink-0" data-popup>
            <button
              onClick={() => {
                setShowDocs((prev) => !prev);
                setShowUserMenu(false);
                setShowGmail(false);
                setConfirmDelete(null);
              }}
              aria-expanded={showDocs}
              className="flex items-center gap-1.5 rounded-full border border-[#E6E0D2] bg-white px-3 py-1.5 text-xs font-medium text-[#6B6B6B] shadow-sm transition hover:border-[#B8CFB3] hover:text-[#5E7C5C]"
            >
              <FileIcon />
              <span className="hidden sm:inline">My documents</span>
              <span className="rounded-full bg-[#EEF4EC] px-1.5 py-px text-[11px] font-semibold text-[#5E7C5C]">
                {uploadedDocs.length}
              </span>
            </button>
            {showDocs && (
              <div className="absolute right-0 top-11 z-20 w-80 max-w-[calc(100vw-2rem)] overflow-hidden rounded-2xl border border-[#E6E0D2] bg-white shadow-xl">
                <div className="flex items-baseline justify-between border-b border-[#EFEBE4] px-4 py-2.5">
                  <p className="text-[13px] font-semibold text-[#1C1C1E]">Your documents</p>
                  <p className="text-[11.5px] text-[#9A968C]">
                    {uploadedDocs.length} {uploadedDocs.length === 1 ? "file" : "files"}
                  </p>
                </div>
                <div className="py-1.5">
                {uploadedDocs.length === 0 ? (
                  <p className="px-4 py-3 text-[12.5px] leading-relaxed text-[#9A968C]">
                    No documents yet. Attach a PDF with the + button to add it here.
                  </p>
                ) : (
                  <ul className="max-h-72 overflow-y-auto">
                    {uploadedDocs.map((doc) => (
                      <li
                        key={doc}
                        className="flex items-center justify-between gap-2 px-4 py-2 transition hover:bg-[#F7F5F0]"
                      >
                        <button
                          onClick={() => openDocument(doc)}
                          title={`Open ${doc}`}
                          className="flex min-w-0 items-center gap-1.5 text-left text-[12.5px] text-[#1C1C1E] hover:text-[#5E7C5C] hover:underline"
                        >
                          <span className="flex-shrink-0 text-[#5E7C5C]">
                            <FileIcon />
                          </span>
                          <span className="truncate">{doc}</span>
                        </button>

                        {confirmDelete === doc ? (
                          <span className="flex flex-shrink-0 items-center gap-1">
                            <button
                              onClick={() => handleDeleteDocument(doc)}
                              disabled={deletingDoc === doc}
                              className="rounded-md bg-red-600 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-red-700 disabled:opacity-50"
                            >
                              {deletingDoc === doc ? "Deleting…" : "Delete"}
                            </button>
                            <button
                              onClick={() => setConfirmDelete(null)}
                              disabled={deletingDoc === doc}
                              className="rounded-md border border-[#E6E0D2] px-2 py-0.5 text-[11px] text-[#6B6B6B] hover:bg-white"
                            >
                              Cancel
                            </button>
                          </span>
                        ) : (
                          <button
                            onClick={() => setConfirmDelete(doc)}
                            aria-label={`Delete ${doc}`}
                            title="Delete this document"
                            className="flex-shrink-0 rounded-md p-1 text-[#9A968C] transition hover:bg-red-50 hover:text-red-600"
                          >
                            <TrashIcon />
                          </button>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
                </div>
              </div>
            )}
          </div>
        )}

        {userEmail ? (
          <div className="relative flex-shrink-0" data-popup>
            <button
              onClick={() => {
                setShowUserMenu((prev) => !prev);
                setShowDocs(false);
                setShowGmail(false);
                setConfirmDelete(null);
              }}
              className="flex h-9 w-9 items-center justify-center rounded-full bg-[#5E7C5C] text-[13px] font-semibold text-white shadow-sm ring-2 ring-white transition hover:bg-[#4E684C]"
              title={userEmail}
              aria-label="Account menu"
              aria-expanded={showUserMenu}
            >
              {userInitial}
            </button>
            {showUserMenu && (
              <div className="absolute right-0 top-11 z-10 w-56 overflow-hidden rounded-2xl border border-[#E6E0D2] bg-white py-1.5 shadow-xl">
                <p className="text-[11px] px-4 pt-1.5 text-[#9A968C]">Signed in as</p>
                <p className="mb-1.5 truncate border-b border-[#EFEBE4] px-4 pb-2.5 text-[13px] font-medium text-[#1C1C1E]">
                  {userEmail}
                </p>
                <button
                  onClick={() => {
                    setShowUserMenu(false);
                    handleClearChat();
                  }}
                  className="w-full px-4 py-2 text-left text-[13px] text-[#1C1C1E] transition hover:bg-[#F7F5F0]"
                >
                  Clear chat
                </button>
                <button
                  onClick={() => {
                    setShowUserMenu(false);
                    handleLogout();
                  }}
                  className="w-full px-4 py-2 text-left text-[13px] text-[#1C1C1E] transition hover:bg-[#F7F5F0]"
                >
                  Log out
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-shrink-0 items-center gap-2">
            <button
              onClick={() => {
                setAuthMode("login");
                setShowAuthModal(true);
              }}
              className="rounded-full border border-[#E6E0D2] bg-white px-4 py-1.5 text-[13px] font-medium text-[#1C1C1E] transition hover:border-[#B8CFB3] hover:text-[#5E7C5C]"
            >
              Log in
            </button>
            <button
              onClick={() => {
                setAuthMode("signup");
                setShowAuthModal(true);
              }}
              className="rounded-full bg-[#5E7C5C] px-4 py-1.5 text-[13px] font-medium text-white shadow-sm transition hover:bg-[#4E684C]"
            >
              Sign up
            </button>
          </div>
        )}
      </header>

      {gmailNotice && (
        <div
          role="status"
          className={`flex items-center justify-center gap-3 border-b px-4 py-2 text-[12.5px] ${
            gmailNotice.ok
              ? "border-[#D7E5D3] bg-[#EEF4EC] text-[#3F5A3D]"
              : "border-red-100 bg-red-50 text-red-700"
          }`}
        >
          <span>{gmailNotice.text}</span>
          <button
            onClick={() => setGmailNotice(null)}
            aria-label="Close"
            className="rounded px-1 text-current opacity-60 transition hover:opacity-100"
          >
            ×
          </button>
        </div>
      )}

      {/* ── Messages ──────────────────────────────── */}
      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto px-3 py-6 sm:px-4 md:px-8">
          <div className="mx-auto max-w-3xl space-y-6">
            {/* Empty state */}
            {isEmpty && (
              <div className="flex flex-col items-center px-2 pt-10 text-center sm:pt-16">
                <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-[#5E7C5C] text-white shadow-md [&_svg]:h-7 [&_svg]:w-7">
                  <BotIcon />
                </div>
                <h2 className="text-[22px] font-semibold tracking-tight text-[#1C1C1E] sm:text-[26px]">
                  Ask questions about your documents
                </h2>
                <p className="mt-2 max-w-md text-[14.5px] leading-relaxed text-[#6B6B6B]">
                  Add a PDF, ask in your own words, and every answer points to the page it came from.
                </p>
              </div>
            )}

            {/* Placeholder while the saved chat is loading */}
            {chatLoading && (
              <div className="space-y-5" aria-label="Loading your chat">
                <div className="flex justify-end">
                  <div className="h-11 w-52 animate-pulse rounded-2xl rounded-br-sm bg-[#E6E0D2]" />
                </div>
                <div className="flex items-end gap-2.5">
                  <div className="h-7 w-7 flex-shrink-0 animate-pulse rounded-full bg-[#E6E0D2]" />
                  <div className="h-20 w-2/3 animate-pulse rounded-2xl rounded-bl-sm bg-[#E6E0D2]" />
                </div>
              </div>
            )}

            {/* Chat messages */}
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`msg-enter flex items-end gap-2.5 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {/* Bot avatar */}
                {msg.role === "assistant" && (
                  <div className="mb-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-[#5E7C5C] text-white shadow-sm">
                    <BotIcon />
                  </div>
                )}

                {msg.uploads ? (
                  <div className="flex w-full max-w-sm flex-col gap-2">
                    {msg.uploads.map((u) => (
                      <div
                        key={u.name}
                        className="flex items-center gap-3 rounded-xl border border-[#E0DBD3] bg-white px-3 py-2.5 shadow-sm"
                      >
                        <span className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg bg-red-50 text-[10px] font-bold text-red-600">
                          PDF
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-[13px] font-medium text-[#1C1C1E]">{u.name}</span>
                          {u.status === "reading" && (
                            <>
                              <span className="mt-0.5 flex items-center gap-1.5 text-[11.5px] text-[#6B6B6B]">
                                <span className="h-3 w-3 animate-spin rounded-full border-2 border-[#B8CFB3] border-t-[#5E7C5C]" />
                                Reading…
                              </span>
                              <span className="mt-1.5 block h-1 w-full animate-pulse rounded-full bg-[#B8CFB3]" />
                            </>
                          )}
                          {u.status === "ready" && (
                            <span className="mt-0.5 flex items-center gap-1 text-[11.5px] text-[#5E7C5C]">
                              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="20 6 9 17 4 12" />
                              </svg>
                              {u.note ?? "Ready"}
                            </span>
                          )}
                          {u.status === "failed" && (
                            <span className="mt-0.5 block text-[11.5px] text-red-600">
                              Couldn&apos;t read · {u.note}
                            </span>
                          )}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : msg.attachments && msg.attachments.length > 0 ? (
                  <div className="flex max-w-[80%] flex-col items-end gap-1.5">
                    {msg.attachments.map((name) => (
                      <button
                        key={name}
                        onClick={() => openDocument(name)}
                        title={`Open ${name}`}
                        className="flex max-w-full items-center gap-2.5 rounded-xl border border-[#E0DBD3] bg-white px-3 py-2 text-left shadow-sm transition hover:border-[#5E7C5C]"
                      >
                        <span className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-red-50 text-[10px] font-bold text-red-600">
                          PDF
                        </span>
                        <span className="min-w-0">
                          <span className="block truncate text-[13px] font-medium text-[#1C1C1E]">
                            {name}
                          </span>
                          <span className="block text-[11px] text-[#9A968C]">Attached · click to open</span>
                        </span>
                      </button>
                    ))}
                    <div className="max-w-full rounded-2xl rounded-br-sm bg-[#5E7C5C] px-4 py-3 text-[14.5px] leading-relaxed text-white shadow-sm">
                      {renderContent(msg.content)}
                    </div>
                  </div>
                ) : (
                <div
                  className={`${
                    msg.structured || parseBlocks(msg.content).some((b) => b.kind === "table")
                      ? "w-full min-w-0 max-w-full"
                      : "max-w-[80%]"
                  } rounded-2xl px-4 py-3 text-[14.5px] leading-[1.65] shadow-sm ${
                    msg.role === "user"
                      ? "rounded-br-sm bg-[#5E7C5C] text-white"
                      : "rounded-bl-sm border border-[#E9E4DA] bg-white text-[#1C1C1E]"
                  }`}
                >
                  {msg.structured ? (
                    <StructuredView structured={msg.structured} />
                  ) : (
                    renderContent(msg.content)
                  )}

                  {/* The emails an answer is about (Gmail) */}
                  {msg.role === "assistant" && msg.emails && msg.emails.length > 0 && (
                    <EmailCards emails={msg.emails} onOpen={(name) => openDocument(name)} />
                  )}

                  {/* Which document(s) the answer came from */}
                  {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2.5 flex flex-wrap items-center gap-1.5 border-t border-[#EFEBE4] pt-2">
                      <span className="text-[11px] font-medium text-[#9A968C]">Sources</span>
                      {msg.sources.map((src) => {
                        const pages =
                          msg.sourceRefs?.find((r) => r.file === src)?.pages ?? [];
                        const pageText = pages.length
                          ? ` · p. ${pages.join(", ")}`
                          : "";
                        return (
                          <button
                            key={src}
                            onClick={() => openDocument(src, pages[0])}
                            title={`Open ${src}${pages.length ? ` at page ${pages[0]}` : ""}`}
                            className="flex max-w-[260px] items-center gap-1 rounded-full border border-[#B8CFB3] bg-[#EEF4EC] px-2 py-0.5 text-[11px] font-medium text-[#5E7C5C] transition hover:border-[#5E7C5C] hover:bg-[#DDEAD9]"
                          >
                            <span className="flex-shrink-0">
                              <FileIcon />
                            </span>
                            <span className="truncate">{src}</span>
                            {pageText && (
                              <span className="flex-shrink-0 text-[#84A282]">{pageText}</span>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {/* The steps behind the answer: intent, searches, model calls */}
                  {msg.role === "assistant" && msg.trace && <AnswerTrace trace={msg.trace} />}
                </div>
                )}

                {/* User avatar */}
                {msg.role === "user" && (
                  <div className="mb-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-[#F6E2E9] text-[#8A4F66] text-[11px] font-bold">
                    {userInitial}
                  </div>
                )}
              </div>
            ))}

            {/* Typing indicator */}
            {loading && (
              <div className="msg-enter flex items-end gap-2.5 justify-start">
                <div className="mb-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-[#5E7C5C] text-white shadow-sm">
                  <BotIcon />
                </div>
                <div className="rounded-2xl rounded-bl-sm border border-[#E0DBD3] bg-white px-4 py-3 shadow-sm">
                  {busyLabel ? (
                    // while a document is really being read: just the words, with a moving shine
                    <span className="shimmer-text text-[13.5px] font-medium">{busyLabel}</span>
                  ) : (
                    // otherwise (thinking about the answer): just the three dots
                    <span className="flex items-center gap-1">
                      <span className="typing-dot inline-block h-2 w-2 rounded-full bg-[#9A968C]" />
                      <span className="typing-dot inline-block h-2 w-2 rounded-full bg-[#9A968C]" />
                      <span className="typing-dot inline-block h-2 w-2 rounded-full bg-[#9A968C]" />
                    </span>
                  )}
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ── Input bar ─────────────────────────────── */}
        <div className="border-t border-[#E6E0D2] bg-[#FEFAF3] px-3 pb-3 pt-3 sm:px-4 sm:pb-4 md:px-8">
          <div className="mx-auto max-w-3xl">
            <div
              className="rounded-2xl border border-[#E6E0D2] bg-white px-3 py-2 shadow-sm transition-all duration-150 focus-within:border-[#5E7C5C] focus-within:shadow-[0_0_0_4px_rgba(94,124,92,0.12)]"
            >
              {/* PDFs attached to the next question */}
              {attachments.length > 0 && (
                <div className="flex flex-wrap gap-2 pb-2 pt-1">
                  {attachments.map((file, i) => (
                    <div
                      key={`${file.name}-${i}`}
                      className="flex max-w-[240px] items-center gap-2 rounded-xl border border-[#E0DBD3] bg-[#FAF8F5] py-1.5 pl-2 pr-1.5"
                    >
                      <span className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md bg-red-50 text-[9px] font-bold text-red-600">
                        PDF
                      </span>
                      <span className="truncate text-[12.5px] text-[#1C1C1E]" title={file.name}>
                        {file.name}
                      </span>
                      <button
                        onClick={() =>
                          setAttachments((prev) => prev.filter((_, idx) => idx !== i))
                        }
                        aria-label={`Remove ${file.name}`}
                        className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full text-[#9A968C] hover:bg-[#EFEBE4] hover:text-[#1C1C1E]"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              )}
              <div className="flex items-center gap-2">
              <label
                title={
                  attachments.length >= MAX_ATTACHMENTS
                    ? `You can attach up to ${MAX_ATTACHMENTS} PDFs at once`
                    : "Attach PDFs"
                }
                onClick={(e) => {
                  // not logged in: open the login popup instead of the file picker
                  if (!requireLogin()) e.preventDefault();
                }}
                className={`flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl text-[#6B6B6B] transition ${
                  loading || attachments.length >= MAX_ATTACHMENTS
                    ? "cursor-not-allowed opacity-40"
                    : "cursor-pointer hover:bg-[#EEF4EC] hover:text-[#5E7C5C]"
                }`}
              >
                <input
                  type="file"
                  accept=".pdf"
                  multiple
                  onChange={handleAttach}
                  disabled={loading || attachments.length >= MAX_ATTACHMENTS}
                  className="hidden"
                />
                <PlusIcon />
              </label>
              <input
                ref={inputRef}
                type="text"
                name="chat-message"
                autoComplete="off"
                data-lpignore="true"
                data-form-type="other"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) =>
                  e.key === "Enter" && !e.shiftKey && handleSend()
                }
                placeholder={
                  loading
                    ? "Waiting for the previous answer…"
                    : attachments.length > 0
                      ? "Ask something about the attached file, or just send to add it…"
                      : uploadedDocs.length > 0
                        ? "Type your question…"
                        : "Type a message, or attach a PDF with +"
                }
                className="flex-1 bg-transparent py-1.5 text-[14.5px] text-[#1C1C1E] placeholder-[#B0ABA2] outline-none disabled:cursor-not-allowed"
              />
              <button
                onClick={handleSend}
                disabled={loading || (!input.trim() && attachments.length === 0)}
                aria-label="Send message"
                className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-[#5E7C5C] text-white shadow-sm transition hover:bg-[#4E684C] active:scale-95 disabled:cursor-not-allowed disabled:bg-[#C9D5C6] disabled:shadow-none"
              >
                <SendIcon />
              </button>
              </div>
            </div>
            <p className="mt-2 text-center text-[11px] text-[#B8B3A8]">
              Answers come only from your uploaded documents. Check the sources for important details.
            </p>
          </div>
        </div>
      </main>

      {showAuthModal && (
        <AuthModal
          onClose={() => setShowAuthModal(false)}
          onAuthSuccess={handleAuthSuccess}
          initialMode={authMode}
        />
      )}
    </div>
  );
}