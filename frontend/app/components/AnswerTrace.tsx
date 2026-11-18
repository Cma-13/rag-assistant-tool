"use client";

import { useState } from "react";

// ── Shape of the trace the backend sends with each answer ──
interface Hit {
  document: string;
  page: number | null;
  similarity: number;
  text: string;
  rrf?: number;
  found_by?: "both" | "meaning" | "keywords";
  vector_rank?: number | null;
  keyword_rank?: number | null;
}

export interface TraceStep {
  kind: "intent" | "rewrite" | "retrieval" | "retrieval_docs" | "agent" | "llm" | "context" | "document" | "note";
  title: string;
  detail?: Record<string, unknown>;
  ms?: number;
}

export interface Trace {
  steps: TraceStep[];
  total_ms: number;
}

const KIND_LABEL: Record<TraceStep["kind"], string> = {
  intent: "Intent",
  rewrite: "Rewrite",
  retrieval: "Retrieval",
  retrieval_docs: "Retrieval",
  agent: "Agent",
  llm: "Language model",
  context: "Context",
  document: "Indexing",
  note: "Step",
};

function formatMs(ms?: number) {
  if (ms === undefined) return "";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`;
}

function where(hit: Hit) {
  return hit.page ? `${hit.document} · p. ${hit.page}` : hit.document;
}

// e5 similarities always sit between about 0.7 and 1.0, so a raw bar would look
// almost full for every passage. The bar is drawn relative to the best match in
// this list (from a floor of 0.7), and the raw value is shown next to it.
const E5_FLOOR = 0.7;

function SimilarityBar({ value, best }: { value: number; best: number }) {
  const span = Math.max(best - E5_FLOOR, 0.01);
  const pct = Math.max(0, Math.min(1, (value - E5_FLOOR) / span)) * 100;
  return (
    <span
      className="flex items-center gap-1.5"
      title={`Cosine similarity ${value.toFixed(3)}. The bar compares it with the best match in this list.`}
    >
      <span className="h-1.5 w-12 overflow-hidden rounded-full bg-[#ECE7DD]">
        <span className="block h-full rounded-full bg-[#5E7C5C]" style={{ width: `${pct}%` }} />
      </span>
      <span className="tabular-nums text-[#6B6B6B]">{value.toFixed(2)}</span>
    </span>
  );
}

function HitList({ hits, showFoundBy }: { hits: Hit[]; showFoundBy?: boolean }) {
  if (!hits.length) return <p className="text-[#9A968C]">Nothing matched.</p>;
  const best = Math.max(...hits.map((h) => h.similarity));
  return (
    <ol className="space-y-1">
      {hits.map((h, i) => (
        <li key={i} className="rounded-lg bg-[#FAF8F4] px-2 py-1.5">
          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-0.5">
            <span className="font-medium text-[#1C1C1E]">
              {i + 1}. {where(h)}
            </span>
            <span className="flex items-center gap-2">
              {showFoundBy && h.found_by && (
                <span
                  className={`whitespace-nowrap rounded-full px-1.5 py-px text-[10.5px] font-medium ${
                    h.found_by === "both"
                      ? "bg-[#EEF4EC] text-[#4E684C]"
                      : "bg-[#F4F1EA] text-[#6B6B6B]"
                  }`}
                >
                  {h.found_by === "both" ? "meaning + keywords" : `${h.found_by} only`}
                </span>
              )}
              {showFoundBy && h.rrf !== undefined && (
                <span className="tabular-nums text-[#6B6B6B]" title="Reciprocal Rank Fusion score">
                  RRF {h.rrf.toFixed(4)}
                </span>
              )}
              <SimilarityBar value={h.similarity} best={best} />
            </span>
          </div>
          <p className="mt-0.5 line-clamp-2 text-[#6B6B6B]">{h.text}</p>
        </li>
      ))}
    </ol>
  );
}

function KeyValues({ detail, skip = [] }: { detail: Record<string, unknown>; skip?: string[] }) {
  const rows = Object.entries(detail).filter(([k, v]) => !skip.includes(k) && v !== null && v !== "");
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-[#9A968C]">{k.replace(/_/g, " ")}</dt>
          <dd className="break-words text-[#1C1C1E]">
            {Array.isArray(v)
              ? v.join(", ")
              : typeof v === "boolean"
                ? v
                  ? "yes"
                  : "no"
                : v && typeof v === "object"
                  ? Object.entries(v as Record<string, unknown>)
                      .map(([name, value]) => `${name}: ${String(value)}`)
                      .join(" · ")
                  : String(v)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function RetrievalDetail({ detail }: { detail: Record<string, unknown> }) {
  const [tab, setTab] = useState<"merged" | "vector" | "keyword">("merged");
  const vector = detail.vector as { method: string; candidates: number; ms: number; top: Hit[] };
  const keyword = detail.keyword as { method: string; query: string; candidates: number; ms: number; top: Hit[] };
  const merged = detail.merged as { method: string; kept: number; results: Hit[] };
  const embedding = detail.embedding as { model: string; dimensions: number; ms: number };
  const scope = detail.scope as string | string[];

  const tabs = [
    { id: "merged", label: `Final ranking (${merged.kept})` },
    { id: "vector", label: `By meaning (${vector.candidates})` },
    { id: "keyword", label: `By keywords (${keyword.candidates})` },
  ] as const;

  return (
    <div className="space-y-2">
      <p className="text-[#6B6B6B]">
        Searched in <span className="text-[#1C1C1E]">{Array.isArray(scope) ? scope.join(", ") : scope}</span>. The
        question was turned into a {embedding.dimensions}-number vector with {embedding.model} (
        {formatMs(embedding.ms)}), then searched two ways and merged.
      </p>
      <div className="flex flex-wrap gap-1" role="tablist">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition ${
              tab === t.id
                ? "border-[#5E7C5C] bg-[#5E7C5C] text-white"
                : "border-[#E6E0D2] bg-white text-[#6B6B6B] hover:border-[#B8CFB3] hover:text-[#5E7C5C]"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      {tab === "merged" && (
        <>
          <p className="text-[#6B6B6B]">
            {merged.method}: passages found by both searches rank higher. Top {merged.kept} kept.
          </p>
          <HitList hits={merged.results} showFoundBy />
          {typeof detail.following_chunks_added === "number" && (
            <p className="text-[#6B6B6B]">
              + {detail.following_chunks_added} following{" "}
              {detail.following_chunks_added === 1 ? "chunk was" : "chunks were"} added, so a fact cut at a chunk
              boundary is not lost.
            </p>
          )}
          <p className="text-[11px] text-[#9A968C]">
            Bars compare each passage with the best match. With this embedding model every passage scores about
            0.7 or more, so only the order matters, not the number itself.
          </p>
        </>
      )}
      {tab === "vector" && (
        <>
          <p className="text-[#6B6B6B]">
            {vector.method}, closest in meaning first ({formatMs(vector.ms)}).
          </p>
          <HitList hits={vector.top} />
        </>
      )}
      {tab === "keyword" && (
        <>
          <p className="text-[#6B6B6B]">
            {keyword.method} for <code className="rounded bg-[#F4F1EA] px-1 font-mono text-[11px]">{keyword.query || "(no keywords)"}</code> (
            {formatMs(keyword.ms)}).
          </p>
          <HitList hits={keyword.top} />
        </>
      )}
    </div>
  );
}

function StepDetail({ step }: { step: TraceStep }) {
  const d = step.detail ?? {};
  switch (step.kind) {
    case "retrieval":
      return <RetrievalDetail detail={d} />;
    case "retrieval_docs": {
      const docs = (d.documents as { file: string; chunks: number; pages: number[] }[]) ?? [];
      return (
        <div className="space-y-1 text-[#6B6B6B]">
          <p>{String(d.method)}</p>
          {docs.length ? (
            <ul className="space-y-0.5">
              {docs.map((doc) => (
                <li key={doc.file}>
                  <span className="text-[#1C1C1E]">{doc.file}</span> · {doc.chunks}{" "}
                  {doc.chunks === 1 ? "chunk" : "chunks"}
                  {doc.pages.length ? ` · page ${doc.pages.join(", ")}` : ""}
                </li>
              ))}
            </ul>
          ) : (
            <p>No document contains it.</p>
          )}
        </div>
      );
    }
    case "rewrite":
      return (
        <p className="text-[#6B6B6B]">
          <span className="text-[#1C1C1E]">&ldquo;{String(d.from)}&rdquo;</span> →{" "}
          <span className="text-[#1C1C1E]">&ldquo;{String(d.to)}&rdquo;</span>
        </p>
      );
    case "intent":
      return (
        <p className="text-[#6B6B6B]">
          Decided by <span className="text-[#1C1C1E]">{String(d.method)}</span>
          {d.final_action ? (
            <>
              , then handled as <span className="text-[#1C1C1E]">{String(d.final_action)}</span>
            </>
          ) : null}
          . Files selected: <span className="text-[#1C1C1E]">{String(d.selected_files)}</span>
        </p>
      );
    case "llm":
      return (
        <p className="text-[#6B6B6B]">
          {String(d.provider)} · <span className="text-[#1C1C1E]">{String(d.model)}</span>
          {d.fallback ? " · used as a fallback because Groq was unavailable" : ""}
        </p>
      );
    case "agent":
      return (
        <p className="text-[#6B6B6B]">
          Tool <code className="rounded bg-[#F4F1EA] px-1 font-mono text-[11px]">{String(d.tool)}</code> with query{" "}
          <span className="text-[#1C1C1E]">&ldquo;{String(d.query)}&rdquo;</span> in{" "}
          <span className="text-[#1C1C1E]">{Array.isArray(d.source_file) ? d.source_file.join(", ") : String(d.source_file)}</span>
          {d.model_wrote ? (
            <span className="text-[#9A968C]"> (the model wrote &ldquo;{String(d.model_wrote)}&rdquo;, matched to the real file)</span>
          ) : null}
        </p>
      );
    case "context":
      return (
        <p className="text-[#6B6B6B]">
          {String(d.passages)} passages from {(d.documents as string[]).join(", ")} · about{" "}
          {String(d.approx_tokens)} tokens ({String(d.characters)} characters)
        </p>
      );
    default:
      return <KeyValues detail={d} />;
  }
}

export default function AnswerTrace({ trace }: { trace: Trace }) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<number | null>(null);
  if (!trace?.steps?.length) return null;

  const searches = trace.steps.filter((s) => s.kind === "retrieval").length;
  const modelCalls = trace.steps.filter((s) => s.kind === "llm").length;

  return (
    <div className="mt-2.5 border-t border-[#EFEBE4] pt-2 text-[12px] leading-relaxed">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-1.5 text-left text-[11.5px] font-medium text-[#6B6B6B] transition hover:text-[#5E7C5C]"
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className={`transition-transform ${open ? "rotate-90" : ""}`}
          aria-hidden="true"
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        How this answer was found
        <span className="font-normal text-[#9A968C]">
          · {searches > 0 ? `${searches} ${searches === 1 ? "search" : "searches"} · ` : ""}
          {modelCalls} model {modelCalls === 1 ? "call" : "calls"} · {formatMs(trace.total_ms)}
        </span>
      </button>

      {open && (
        <ol className="relative mt-2.5 space-y-2 pl-5 before:absolute before:bottom-2 before:left-[7px] before:top-2 before:w-px before:bg-[#E6E0D2]">
          {trace.steps.map((step, i) => {
            const isOpen = expanded === i || step.kind !== "retrieval";
            return (
              <li key={i} className="relative">
                <span
                  className={`absolute -left-5 top-[3px] flex h-[15px] w-[15px] items-center justify-center rounded-full border-2 bg-white ${
                    step.kind === "retrieval" || step.kind === "retrieval_docs" || step.kind === "agent"
                      ? "border-[#5E7C5C]"
                      : step.kind === "llm"
                        ? "border-[#C98AA2]"
                        : "border-[#CFC8B8]"
                  }`}
                  aria-hidden="true"
                />
                <div className="flex flex-wrap items-baseline justify-between gap-x-2">
                  <p>
                    <span className="mr-1.5 text-[10.5px] font-semibold text-[#9A968C]">{KIND_LABEL[step.kind]}</span>
                    <span className="font-medium text-[#1C1C1E]">{step.title}</span>
                  </p>
                  {step.ms !== undefined && (
                    <span className="tabular-nums text-[11px] text-[#9A968C]">{formatMs(step.ms)}</span>
                  )}
                </div>
                {step.kind === "retrieval" && (
                  <button
                    type="button"
                    onClick={() => setExpanded(expanded === i ? null : i)}
                    className="mt-0.5 text-[11px] font-medium text-[#5E7C5C] underline-offset-2 hover:underline"
                  >
                    {expanded === i ? "Hide search results" : "Show search results"}
                  </button>
                )}
                {isOpen && step.detail && (
                  <div className="mt-1">
                    <StepDetail step={step} />
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
