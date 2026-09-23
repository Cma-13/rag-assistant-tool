"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
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
  const [uploading, setUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadedDocs, setUploadedDocs] = useState<string[]>([]);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/documents")
      .then((res) => res.json())
      .then((data) => setUploadedDocs(data.documents ?? []))
      .catch(() => {});
  }, []);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input }), // no source_file — searches whole knowledge base
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I couldn't reach the server. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const processFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Only PDF files are supported. Please upload a valid PDF.",
        },
      ]);
      return;
    }
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (data.chunks_stored === 0) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              data.message ??
              `"${file.name}" is already in the knowledge base - skipped.`,
          },
        ]);
      } else {
        setUploadedDocs((prev) => [...new Set([...prev, data.filename])]);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Added "${data.filename}" to the knowledge base.`,
          },
        ]);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "I couldn't process that file." },
      ]);
    } finally {
      setUploading(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    await processFiles(Array.from(files));
    e.target.value = "";
  };

  const processFiles = async (files: File[]) => {
    for (let i = 0; i < files.length; i++) {
      setUploadProgress(
        files.length > 1 ? `Uploading ${i + 1} of ${files.length}…` : null,
      );
      await processFile(files[i]);
    }
    setUploadProgress(null);
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      await processFiles(Array.from(files));
    }
  };

  // Render **bold** markdown inline
  const renderContent = (text: string) =>
    text
      .split(/(\*\*[^*]+\*\*)/)
      .map((part, i) =>
        part.startsWith("**") && part.endsWith("**") ? (
          <strong key={i}>{part.slice(2, -2)}</strong>
        ) : (
          <span key={i}>{part}</span>
        ),
      );

  const isEmpty = messages.length === 0;

  return (
    <div
      className="flex h-screen flex-col bg-[#F0EDE8]"
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      {/* Drag-and-drop overlay */}
      {isDragging && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#2D6A6A]/10 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-3 rounded-2xl border-2 border-dashed border-[#2D6A6A] bg-white/80 px-12 py-10 text-[#2D6A6A]">
            <UploadIcon />
            <p className="text-base font-semibold">Drop your PDF here</p>
          </div>
        </div>
      )}

      {/* ── Top bar ───────────────────────────────── */}
      <header className="flex items-center justify-between gap-4 border-b border-[#DDD9D0] bg-[#F0EDE8] px-6 py-3.5">
        {/* Brand */}
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-[#2D6A6A] text-white shadow-sm">
            <BotIcon />
          </div>
          <div className="min-w-0">
            <h1 className="text-sm font-semibold text-[#1C1C1E] leading-tight">
              DocQuery
            </h1>
            <p className="text-[11px] text-[#9A968C] leading-tight">
              RAG Assistant
            </p>
          </div>
        </div>

        {/* Knowledge base badge */}
        {uploadedDocs.length > 0 && (
          <div className="flex min-w-0 flex-1 items-center justify-center">
            <div className="flex max-w-xs items-center gap-1.5 rounded-full border border-[#C4DFE0] bg-[#EAF3F3] px-3 py-1">
              <span className="text-[#2D6A6A] flex-shrink-0">
                <FileIcon />
              </span>
              <span className="truncate text-[11px] font-medium text-[#2D6A6A]">
                {uploadedDocs.length} document
                {uploadedDocs.length > 1 ? "s" : ""} in knowledge base
              </span>
            </div>
          </div>
        )}

        {/* Upload button */}
        <label className="flex flex-shrink-0 cursor-pointer items-center gap-1.5 rounded-full border border-[#DDD9D0] bg-white px-3.5 py-1.5 text-xs font-medium text-[#6B6B6B] shadow-sm transition hover:border-[#C4DFE0] hover:bg-[#EAF3F3] hover:text-[#2D6A6A]">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            multiple
            onChange={handleUpload}
            disabled={uploading}
            className="hidden"
          />
          {uploading ? (
            <span className="uploading-pulse flex items-center gap-1.5">
              <UploadIcon /> {uploadProgress ?? "Processing…"}
            </span>
          ) : (
            <span className="flex items-center gap-1.5">
              <UploadIcon /> Upload PDFs
            </span>
          )}
        </label>
      </header>

      {/* ── Messages ──────────────────────────────── */}
      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8">
          <div className="mx-auto max-w-2xl space-y-5">
            {/* Empty state */}
            {isEmpty && (
              <div className="flex flex-col items-center justify-center py-20 text-center">
                <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl border border-[#DDD9D0] bg-white text-[#2D6A6A] shadow-sm">
                  <BotIcon />
                </div>
                <h2 className="mb-1.5 text-base font-semibold text-[#1C1C1E]">
                  Ready to explore your knowledge base
                </h2>
                <p className="max-w-xs text-sm leading-relaxed text-[#9A968C]">
                  Upload one or more PDF documents to get started.
                </p>
                <div className="mt-7 flex flex-wrap justify-center gap-2">
                  {["Context-aware answers", "PDF documents"].map((feat) => (
                    <span
                      key={feat}
                      className="rounded-full border border-[#DDD9D0] bg-white px-3 py-1 text-xs text-[#6B6B6B]"
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
                  <div className="mb-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-[#2D6A6A] text-white shadow-sm">
                    <BotIcon />
                  </div>
                )}

                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-3 text-[14.5px] leading-relaxed shadow-sm ${
                    msg.role === "user"
                      ? "rounded-br-sm bg-[#2D6A6A] text-white"
                      : "rounded-bl-sm border border-[#E0DBD3] bg-white text-[#1C1C1E]"
                  }`}
                >
                  {renderContent(msg.content)}
                </div>

                {/* User avatar */}
                {msg.role === "user" && (
                  <div className="mb-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-[#D8D3C9] text-[#6B6B6B] text-[11px] font-bold">
                    U
                  </div>
                )}
              </div>
            ))}

            {/* Typing indicator */}
            {loading && (
              <div className="msg-enter flex items-end gap-2.5 justify-start">
                <div className="mb-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-[#2D6A6A] text-white shadow-sm">
                  <BotIcon />
                </div>
                <div className="rounded-2xl rounded-bl-sm border border-[#E0DBD3] bg-white px-4 py-3 shadow-sm">
                  <span className="flex items-center gap-1">
                    <span className="typing-dot inline-block h-2 w-2 rounded-full bg-[#9A968C]" />
                    <span className="typing-dot inline-block h-2 w-2 rounded-full bg-[#9A968C]" />
                    <span className="typing-dot inline-block h-2 w-2 rounded-full bg-[#9A968C]" />
                  </span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ── Input bar ─────────────────────────────── */}
        <div className="border-t border-[#DDD9D0] bg-[#F0EDE8] px-4 py-4 md:px-8">
          <div className="mx-auto max-w-2xl">
            <div
              className={`flex items-center gap-2 rounded-2xl border bg-white px-4 py-2 shadow-sm transition-all duration-150 ${
                uploadedDocs.length > 0
                  ? "border-[#DDD9D0] focus-within:border-[#2D6A6A] focus-within:shadow-[0_0_0_3px_rgba(45,106,106,0.10)]"
                  : "border-[#DDD9D0] opacity-60"
              }`}
            >
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) =>
                  e.key === "Enter" && !e.shiftKey && handleSend()
                }
                placeholder={
                  uploading
                    ? "Processing document…"
                    : loading
                      ? "Waiting for the previous answer…"
                      : uploadedDocs.length > 0
                        ? "Ask anything about your knowledge base…"
                        : "Say hi, or upload a PDF to ask about documents…"
                }
                disabled={uploading}
                className="flex-1 bg-transparent py-1.5 text-[14.5px] text-[#1C1C1E] placeholder-[#B0ABA2] outline-none disabled:cursor-not-allowed"
              />
              <button
                onClick={handleSend}
                disabled={uploading || loading || !input.trim()}
                aria-label="Send message"
                className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl bg-[#2D6A6A] text-white shadow-sm transition hover:bg-[#255757] active:scale-95 disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <SendIcon />
              </button>
            </div>
            <p className="mt-2 text-center text-[11px] text-[#B8B3A8]">
              Answers are based on your uploaded knowledge base.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
