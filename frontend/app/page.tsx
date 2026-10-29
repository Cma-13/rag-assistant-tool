"use client";

import { useState, useRef, useEffect } from "react";
import AuthModal from "./components/AuthModal";

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
  attachments?: string[]; // PDFs attached to this question (shown as file cards)
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

      <div className="max-h-[420px] overflow-auto rounded-lg border border-[#E0DBD3]">
        <table className="w-auto min-w-[55%] table-auto border-collapse text-[12.5px]">
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
    <div className="overflow-hidden rounded-lg border border-[#E0DBD3]">
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
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
      <line x1="12" y1="3" x2="12" y2="7" />
      <circle cx="8.5" cy="16" r="1" fill="currentColor" />
      <circle cx="15.5" cy="16" r="1" fill="currentColor" />
    </svg>
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

  // Pop-ups (My documents, account menu) close when you click anywhere else or press Esc.
  useEffect(() => {
    const closePopups = () => {
      setShowDocs(false);
      setShowUserMenu(false);
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
    fetch("http://127.0.0.1:8000/documents", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => setUploadedDocs(data.documents ?? []))
      .catch(() => {});
  }, [token]);

  // ── Chat history: saved on the server, one history per user ──
  // `chatReadyFor` is the user whose saved chat has been loaded. Nothing is
  // saved before that, so a failed load can never overwrite the real history.
  const localChatKey = userEmail ? `chatHistory:${userEmail}` : null;
  const [chatReadyFor, setChatReadyFor] = useState<string | null>(null);

  // When a user is logged in (or logs in again), bring back their saved chat.
  useEffect(() => {
    if (!token || !userEmail) {
      setChatReadyFor(null);
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

    fetch("http://127.0.0.1:8000/chat", {
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
      })
      .catch(() => {
        // Server not reachable: show the browser copy, but do not save anything.
        if (!cancelled) setMessages(readLocal());
      });
    return () => {
      cancelled = true;
    };
  }, [token, userEmail]);

  // Save the chat shortly after it changes (only the last 100 messages are kept).
  useEffect(() => {
    if (!token || !userEmail || chatReadyFor !== userEmail) return;
    const timer = setTimeout(() => {
      fetch("http://127.0.0.1:8000/chat", {
        method: "PUT",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ messages: messages.slice(-100) }),
      }).catch(() => {
        /* saving failed - the chat still works, it just won't be saved this time */
      });
    }, 800);
    return () => clearTimeout(timer);
  }, [messages, token, userEmail, chatReadyFor]);

  const handleClearChat = () => {
    setMessages([]);
    if (token) {
      fetch("http://127.0.0.1:8000/chat", {
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
        `http://127.0.0.1:8000/documents?name=${encodeURIComponent(name)}`,
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
        `http://127.0.0.1:8000/documents/file?name=${encodeURIComponent(name)}`,
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
      }[] = [];
      for (let i = 0; i < past.length - 1; i++) {
        if (past[i].role === "user" && past[i + 1]?.role === "assistant") {
          recentHistory.push({
            question: past[i].content,
            answer: past[i + 1].content,
            sources: past[i + 1].sources ?? [],
            source_refs: past[i + 1].sourceRefs ?? [],
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

      const res = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers,
        body: JSON.stringify({
          question,
          history: lastFewTurns,
          chat_user_messages: chatUserMessages,
          ...(scope && scope.length ? { source_files: scope } : {}),
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
        },
      ]);
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

  // Upload one PDF without adding chat messages. Returns the name it is stored under.
  const uploadOne = async (
    file: File,
  ): Promise<{ name?: string; already?: boolean; error?: string }> => {
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      const data = await res.json();
      if (!res.ok || !data.filename) {
        return { error: data.detail ?? `I couldn't process "${file.name}".` };
      }
      setUploadedDocs((prev) => [...new Set([...prev, data.filename])]);
      return { name: data.filename, already: data.chunks_stored === 0 };
    } catch {
      return { error: `I couldn't process "${file.name}".` };
    }
  };

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
        // Read the attached files first (they join the knowledge base).
        const stored: string[] = [];
        const added: string[] = [];
        const already: string[] = [];
        for (let i = 0; i < files.length; i++) {
          setBusyLabel(
            files.length > 1
              ? `Reading document ${i + 1} of ${files.length}…`
              : "Reading your document…",
          );
          const result = await uploadOne(files[i]);
          if (result.name) {
            stored.push(result.name);
            (result.already ? already : added).push(result.name);
          } else {
            setMessages((prev) => [
              ...prev,
              { role: "assistant", content: result.error ?? "I couldn't process that file." },
            ]);
          }
        }
        setBusyLabel(null);

        if (!question) {
          // Only files, no question: just confirm they were added.
          const lines: string[] = [];
          if (added.length === 1) lines.push(`Uploaded "${added[0]}"`);
          if (added.length > 1)
            lines.push(`Uploaded ${added.length} documents: ${added.map((n) => `"${n}"`).join(", ")}`);
          already.forEach((n) => lines.push(`"${n}" is already uploaded.`));
          if (lines.length) {
            setMessages((prev) => [...prev, { role: "assistant", content: lines.join("\n") }]);
          }
          return;
        }

        if (stored.length === 0) return; // nothing could be read, nothing to ask about
        scope = [...new Set(stored)];
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
  const attachFiles = (files: File[]) => {
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

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
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
  const isEmpty = messages.length === 0;

  return (
    <div
      className="flex h-screen flex-col bg-[#FEFAF3]"
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
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
      <header className="flex items-center justify-between gap-4 border-b border-[#E6E0D2] bg-[#FEFAF3] px-6 py-3.5">
        {/* Brand */}
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-[#5E7C5C] text-white shadow-sm">
            <BotIcon />
          </div>
          <div className="min-w-0">
            <h1 className="text-sm font-semibold text-[#1C1C1E] leading-tight">
              DocQuery
            </h1>
          </div>
        </div>

        {/* Spacer keeps the upload and account buttons on the right */}
        <div className="flex-1" />

        {/* My documents */}
        {token && (
          <div className="relative flex-shrink-0" data-popup>
            <button
              onClick={() => {
                setShowDocs((prev) => !prev);
                setShowUserMenu(false);
                setConfirmDelete(null);
              }}
              className="flex items-center gap-1.5 rounded-full border border-[#E6E0D2] bg-white px-3.5 py-1.5 text-xs font-medium text-[#6B6B6B] shadow-sm transition hover:border-[#B8CFB3] hover:bg-[#EEF4EC] hover:text-[#5E7C5C]"
            >
              <FileIcon /> My documents ({uploadedDocs.length})
            </button>
            {showDocs && (
              <div className="absolute right-0 top-10 z-20 w-72 rounded-xl border border-[#E6E0D2] bg-white py-2 shadow-lg">
                {uploadedDocs.length === 0 ? (
                  <p className="px-3 py-2 text-[12.5px] text-[#9A968C]">
                    No documents yet. Upload a PDF to get started.
                  </p>
                ) : (
                  <ul className="max-h-72 overflow-y-auto">
                    {uploadedDocs.map((doc) => (
                      <li
                        key={doc}
                        className="flex items-center justify-between gap-2 px-3 py-1.5 hover:bg-[#FBF0F3]"
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
            )}
          </div>
        )}

        {userEmail ? (
          <div className="relative flex-shrink-0" data-popup>
            <button
              onClick={() => {
                setShowUserMenu((prev) => !prev);
                setShowDocs(false);
                setConfirmDelete(null);
              }}
              className="flex h-8 w-8 items-center justify-center rounded-full bg-[#5E7C5C] text-[13px] font-semibold text-white shadow-sm"
              title={userEmail}
            >
              {userInitial}
            </button>
            {showUserMenu && (
              <div className="absolute right-0 top-10 z-10 w-48 rounded-xl border border-[#E6E0D2] bg-white py-2 shadow-lg">
                <p className="truncate px-3 pb-2 text-[12px] text-[#9A968C]">
                  {userEmail}
                </p>
                <button
                  onClick={() => {
                    setShowUserMenu(false);
                    handleClearChat();
                  }}
                  className="w-full px-3 py-1.5 text-left text-[13px] text-[#1C1C1E] hover:bg-[#FBF0F3]"
                >
                  Clear chat
                </button>
                <button
                  onClick={() => {
                    setShowUserMenu(false);
                    handleLogout();
                  }}
                  className="w-full px-3 py-1.5 text-left text-[13px] text-[#1C1C1E] hover:bg-[#FBF0F3]"
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
              className="rounded-xl border border-[#E6E0D2] px-3 py-1.5 text-[13px] font-medium text-[#1C1C1E] hover:bg-[#FBF0F3]"
            >
              Log in
            </button>
            <button
              onClick={() => {
                setAuthMode("signup");
                setShowAuthModal(true);
              }}
              className="rounded-xl bg-[#5E7C5C] px-3 py-1.5 text-[13px] font-medium text-white hover:bg-[#4E684C]"
            >
              Sign up
            </button>
          </div>
        )}
      </header>

      {/* ── Messages ──────────────────────────────── */}
      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8">
          <div className="mx-auto max-w-3xl space-y-5">
            {/* Empty state */}
            {isEmpty && (
              <div className="flex flex-col items-center justify-center py-20 text-center">
                <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl border border-[#EFD3DC] bg-[#F6E2E9] text-[#5E7C5C] shadow-sm">
                  <BotIcon />
                </div>
                <h2 className="mb-1.5 text-base font-semibold text-[#1C1C1E]">
                  What&apos;s on the agenda today?
                </h2>
                <p className="max-w-xs text-sm leading-relaxed text-[#9A968C]">
                  Ask me anything, or attach a PDF with the + button to get started.
                </p>
                <div className="mt-7 flex flex-wrap justify-center gap-2">
                  {["Context-aware answers", "PDF documents"].map((feat) => (
                    <span
                      key={feat}
                      className="rounded-full border border-[#EFD3DC] bg-[#FBEFF3] px-3 py-1 text-xs text-[#8A4F66]"
                    >
                      {feat}
                    </span>
                  ))}
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

                {msg.attachments && msg.attachments.length > 0 ? (
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
                          <span className="block text-[11px] text-[#9A968C]">PDF</span>
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
                  } rounded-2xl px-4 py-3 text-[14.5px] leading-relaxed shadow-sm ${
                    msg.role === "user"
                      ? "rounded-br-sm bg-[#5E7C5C] text-white"
                      : "rounded-bl-sm border border-[#E0DBD3] bg-white text-[#1C1C1E]"
                  }`}
                >
                  {msg.structured ? (
                    <StructuredView structured={msg.structured} />
                  ) : (
                    renderContent(msg.content)
                  )}

                  {/* Which document(s) the answer came from */}
                  {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2.5 flex flex-wrap items-center gap-1.5 border-t border-[#EFEBE4] pt-2">
                      <span className="text-[11px] text-[#9A968C]">Sources:</span>
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
                  <span className="flex items-center gap-1">
                    <span className="typing-dot inline-block h-2 w-2 rounded-full bg-[#9A968C]" />
                    <span className="typing-dot inline-block h-2 w-2 rounded-full bg-[#9A968C]" />
                    <span className="typing-dot inline-block h-2 w-2 rounded-full bg-[#9A968C]" />
                    {busyLabel && (
                      <span className="ml-2 text-[12.5px] text-[#9A968C]">{busyLabel}</span>
                    )}
                  </span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ── Input bar ─────────────────────────────── */}
        <div className="border-t border-[#E6E0D2] bg-[#FEFAF3] px-4 py-4 md:px-8">
          <div className="mx-auto max-w-3xl">
            <div
              className={`rounded-2xl border bg-white px-4 py-2 shadow-sm transition-all duration-150 ${
                uploadedDocs.length > 0 || attachments.length > 0
                  ? "border-[#E6E0D2] focus-within:border-[#5E7C5C] focus-within:shadow-[0_0_0_3px_rgba(45,106,106,0.10)]"
                  : "border-[#E6E0D2] opacity-60"
              }`}
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
                className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl text-[#6B6B6B] transition ${
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
                className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl bg-[#5E7C5C] text-white shadow-sm transition hover:bg-[#4E684C] active:scale-95 disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <SendIcon />
              </button>
              </div>
            </div>
            <p className="mt-2 text-center text-[11px] text-[#B8B3A8]">
              Answers are based on your uploaded knowledge base.
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