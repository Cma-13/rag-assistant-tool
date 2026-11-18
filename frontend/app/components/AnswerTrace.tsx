"use client";

import { createContext, useContext, useState } from "react";

// ── Shape of the trace the backend sends with each answer ──
interface Hit {
  id?: number;
  position?: number | null;
  of?: number | null;
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
  kind:
    | "intent"
    | "rewrite"
    | "retrieval"
    | "retrieval_docs"
    | "agent"
    | "llm"
    | "context"
    | "document"
    | "note";
  title: string;
  detail?: Record<string, unknown>;
  ms?: number;
}

// One chunk's details, stored once per trace and looked up by id
interface ChunkInfo {
  document: string;
  page: number | null;
  position: number | null;
  of: number | null;
  text: string;
}

export interface Trace {
  steps: TraceStep[];
  chunks?: Record<string, ChunkInfo>; // older saved answers do not have this
  total_ms: number;
}

const ChunksContext = createContext<Record<string, ChunkInfo>>({});

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

function where(hit: {
  document: string;
  page?: number | null;
  position?: number | null;
  of?: number | null;
}) {
  let text = hit.document;
  if (hit.page) text += ` · p. ${hit.page}`;
  if (hit.position && hit.of) text += ` · chunk ${hit.position} of ${hit.of}`;
  return text;
}

// "1, 2, 3, 5" -> "1–3, 5"
function pageRanges(pages: number[]) {
  const sorted = [...pages].sort((a, b) => a - b);
  const parts: string[] = [];
  for (let i = 0; i < sorted.length; i++) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j] + 1) j++;
    parts.push(i === j ? `${sorted[i]}` : `${sorted[i]}–${sorted[j]}`);
    i = j;
  }
  return parts.join(", ");
}

function plural(n: number, word: string) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

// The full text of a chunk, shown when a row is opened
function FullText({ text }: { text: string }) {
  return (
    <p className="mt-1.5 max-h-60 overflow-auto whitespace-pre-wrap rounded-md border border-[#EFEBE4] bg-white px-2 py-1.5 text-[#1C1C1E]">
      {text}
    </p>
  );
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
        <span
          className="block h-full rounded-full bg-[#5E7C5C]"
          style={{ width: `${pct}%` }}
        />
      </span>
      <span className="tabular-nums text-[#6B6B6B]">{value.toFixed(2)}</span>
    </span>
  );
}

function HitList({
  hits,
  showFoundBy,
}: {
  hits: Hit[];
  showFoundBy?: boolean;
}) {
  if (!hits.length) return <p className="text-[#9A968C]">Nothing matched.</p>;
  const best = Math.max(...hits.map((h) => h.similarity));
  return (
    <ol className="space-y-1">
      {hits.map((h, i) => (
        <HitRow
          key={i}
          hit={h}
          index={i}
          best={best}
          showFoundBy={showFoundBy}
        />
      ))}
    </ol>
  );
}

function HitRow({
  hit: h,
  index: i,
  best,
  showFoundBy,
}: {
  hit: Hit;
  index: number;
  best: number;
  showFoundBy?: boolean;
}) {
  const chunks = useContext(ChunksContext);
  const [open, setOpen] = useState(false);
  const full = h.id !== undefined ? chunks[String(h.id)]?.text : undefined;
  return (
    <li className="rounded-lg bg-[#FAF8F4] px-2 py-1.5">
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
              {h.found_by === "both"
                ? "meaning + keywords"
                : `${h.found_by} only`}
            </span>
          )}
          {showFoundBy && h.rrf !== undefined && (
            <span
              className="tabular-nums text-[#6B6B6B]"
              title="Reciprocal Rank Fusion score"
            >
              RRF {h.rrf.toFixed(4)}
            </span>
          )}
          <SimilarityBar value={h.similarity} best={best} />
        </span>
      </div>
      {open && full ? (
        <FullText text={full} />
      ) : (
        <p className="mt-0.5 line-clamp-2 text-[#6B6B6B]">{h.text}</p>
      )}
      {full && (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="mt-0.5 text-[11px] font-medium text-[#5E7C5C] underline-offset-2 hover:underline"
        >
          {open ? "Show less" : "Show full chunk"}
        </button>
      )}
    </li>
  );
}

// The exact passages the language model was given, in order
function ContextDetail({ detail: d }: { detail: Record<string, unknown> }) {
  const chunks = useContext(ChunksContext);
  const [show, setShow] = useState(false);
  const ids = (d.ids as number[] | undefined) ?? [];
  const used = new Set((d.likely_used as number[] | undefined) ?? []);
  const summary = (
    <p className="text-[#6B6B6B]">
      {String(d.passages)} passages from {(d.documents as string[]).join(", ")}{" "}
      · about {String(d.approx_tokens)} tokens ({String(d.characters)}{" "}
      characters)
    </p>
  );
  if (!ids.length || !ids.some((id) => chunks[String(id)])) return summary;
  return (
    <div className="space-y-1.5">
      {summary}
      <button
        type="button"
        onClick={() => setShow((s) => !s)}
        aria-expanded={show}
        className="text-[11px] font-medium text-[#5E7C5C] underline-offset-2 hover:underline"
      >
        {show
          ? "Hide the passages"
          : `Show the ${ids.length} passages, in the order the model saw them`}
      </button>
      {show && (
        <>
          <ol className="space-y-1">
            {ids.map((id, i) => (
              <PassageRow
                key={id}
                index={i}
                info={chunks[String(id)]}
                used={used.has(id)}
              />
            ))}
          </ol>
          {used.size > 0 && (
            <p className="text-[11px] text-[#9A968C]">
              &ldquo;Likely used&rdquo; means the passage shares words or
              numbers with the answer. It is an estimate: the model saw every
              passage above and may use one without repeating its words.
            </p>
          )}
        </>
      )}
    </div>
  );
}

function PassageRow({
  index,
  info,
  used,
}: {
  index: number;
  info?: ChunkInfo;
  used: boolean;
}) {
  const [open, setOpen] = useState(false);
  if (!info) return null;
  return (
    <li
      className={`rounded-lg px-2 py-1.5 ${used ? "bg-[#EEF4EC]" : "bg-[#FAF8F4]"}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-x-3">
        <span className="font-medium text-[#1C1C1E]">
          {index + 1}. {where(info)}
        </span>
        {used && (
          <span className="whitespace-nowrap rounded-full bg-[#5E7C5C] px-1.5 py-px text-[10.5px] font-medium text-white">
            likely used
          </span>
        )}
      </div>
      {open ? (
        <FullText text={info.text} />
      ) : (
        <p className="mt-0.5 line-clamp-2 text-[#6B6B6B]">{info.text}</p>
      )}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="mt-0.5 text-[11px] font-medium text-[#5E7C5C] underline-offset-2 hover:underline"
      >
        {open ? "Show less" : "Show full chunk"}
      </button>
    </li>
  );
}

// How one source document was prepared when it was uploaded
function DocumentDetail({
  detail: d,
  showPipeline,
}: {
  detail: Record<string, unknown>;
  showPipeline: boolean;
}) {
  if (!("recorded" in d)) return <KeyValues detail={d} />; // answers saved before this view existed
  const textPages = (d.text_pages as number[] | undefined) ?? [];
  const ocrPages = (d.ocr_pages as number[] | undefined) ?? [];
  const tablePages = (d.table_pages as number[] | undefined) ?? [];
  const tables = Number(d.tables ?? 0);

  let reading: string;
  if (!d.recorded) {
    reading =
      "Not recorded: this file was uploaded before page details were saved. Upload it again to see them.";
  } else if (ocrPages.length && textPages.length) {
    reading = `Page ${pageRanges(textPages)} from the text layer; page ${pageRanges(ocrPages)} read by OCR (scanned)`;
  } else if (ocrPages.length) {
    reading = `${plural(ocrPages.length, "page")} read by OCR (scanned): page ${pageRanges(ocrPages)}`;
  } else {
    reading = `${plural(textPages.length, "page")} read from the text layer (no OCR needed)`;
  }

  const rows: [string, string][] = [
    ["Pages", d.pages ? String(d.pages) : "unknown"],
    ["How pages were read", reading],
  ];
  if (d.recorded) {
    rows.push([
      "Tables found",
      tables
        ? `${plural(tables, "table")} on page ${pageRanges(tablePages)}`
        : "none",
    ]);
  }
  rows.push([
    "Chunks",
    `${String(d.chunks)} (about ${String(d.average_chunk_characters)} characters each` +
      (Number(d.chunks_with_tables)
        ? `, ${String(d.chunks_with_tables)} contain a table)`
        : ")"),
  ]);
  if (d.added) rows.push(["Uploaded", String(d.added)]);

  return (
    <div className="space-y-1">
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-[#9A968C]">{k}</dt>
            <dd className="break-words text-[#1C1C1E]">{v}</dd>
          </div>
        ))}
      </dl>
      {showPipeline && (
        <p className="text-[11px] text-[#9A968C]">
          Same for every document: split at headings into chunks of about 500
          characters (50 overlap), each turned into 768 numbers with
          multilingual-e5-base, stored in pgvector with a full-text index.
        </p>
      )}
    </div>
  );
}

function KeyValues({
  detail,
  skip = [],
}: {
  detail: Record<string, unknown>;
  skip?: string[];
}) {
  const rows = Object.entries(detail).filter(
    ([k, v]) => !skip.includes(k) && v !== null && v !== "",
  );
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
  const vector = detail.vector as {
    method: string;
    candidates: number;
    ms: number;
    top: Hit[];
  };
  const keyword = detail.keyword as {
    method: string;
    query: string;
    candidates: number;
    ms: number;
    top: Hit[];
  };
  const merged = detail.merged as {
    method: string;
    kept: number;
    results: Hit[];
  };
  const embedding = detail.embedding as {
    model: string;
    dimensions: number;
    ms: number;
  };
  const scope = detail.scope as string | string[];

  const tabs = [
    { id: "merged", label: `Final ranking (${merged.kept})` },
    { id: "vector", label: `By meaning (${vector.candidates})` },
    { id: "keyword", label: `By keywords (${keyword.candidates})` },
  ] as const;

  return (
    <div className="space-y-2">
      <p className="text-[#6B6B6B]">
        Searched in{" "}
        <span className="text-[#1C1C1E]">
          {Array.isArray(scope) ? scope.join(", ") : scope}
        </span>
        . The question was turned into a {embedding.dimensions}-number vector
        with {embedding.model} ({formatMs(embedding.ms)}), then searched two
        ways and merged.
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
            {merged.method}: passages found by both searches rank higher. Top{" "}
            {merged.kept} kept.
          </p>
          <HitList hits={merged.results} showFoundBy />
          {typeof detail.following_chunks_added === "number" && (
            <p className="text-[#6B6B6B]">
              + {detail.following_chunks_added} following{" "}
              {detail.following_chunks_added === 1
                ? "chunk was"
                : "chunks were"}{" "}
              added, so a fact cut at a chunk boundary is not lost.
            </p>
          )}
          <p className="text-[11px] text-[#9A968C]">
            Bars compare each passage with the best match. With this embedding
            model every passage scores about 0.7 or more, so only the order
            matters, not the number itself.
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
            {keyword.method} for{" "}
            <code className="rounded bg-[#F4F1EA] px-1 font-mono text-[11px]">
              {keyword.query || "(no keywords)"}
            </code>{" "}
            ({formatMs(keyword.ms)}).
          </p>
          <HitList hits={keyword.top} />
        </>
      )}
    </div>
  );
}

function StepDetail({
  step,
  showPipeline = false,
}: {
  step: TraceStep;
  showPipeline?: boolean;
}) {
  const d = step.detail ?? {};
  switch (step.kind) {
    case "retrieval":
      return <RetrievalDetail detail={d} />;
    case "retrieval_docs": {
      const docs =
        (d.documents as { file: string; chunks: number; pages: number[] }[]) ??
        [];
      return (
        <div className="space-y-1 text-[#6B6B6B]">
          <p>{String(d.method)}</p>
          {docs.length ? (
            <ul className="space-y-0.5">
              {docs.map((doc) => (
                <li key={doc.file}>
                  <span className="text-[#1C1C1E]">{doc.file}</span> ·{" "}
                  {doc.chunks} {doc.chunks === 1 ? "chunk" : "chunks"}
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
          <span className="text-[#1C1C1E]">&ldquo;{String(d.from)}&rdquo;</span>{" "}
          → <span className="text-[#1C1C1E]">&ldquo;{String(d.to)}&rdquo;</span>
        </p>
      );
    case "intent":
      return (
        <p className="text-[#6B6B6B]">
          Decided by <span className="text-[#1C1C1E]">{String(d.method)}</span>
          {d.final_action ? (
            <>
              , then handled as{" "}
              <span className="text-[#1C1C1E]">{String(d.final_action)}</span>
            </>
          ) : null}
          . Files selected:{" "}
          <span className="text-[#1C1C1E]">{String(d.selected_files)}</span>
        </p>
      );
    case "llm":
      return (
        <p className="text-[#6B6B6B]">
          {String(d.provider)} ·{" "}
          <span className="text-[#1C1C1E]">{String(d.model)}</span>
          {d.fallback
            ? " · used as a fallback because Groq was unavailable"
            : ""}
        </p>
      );
    case "agent":
      return (
        <p className="text-[#6B6B6B]">
          Tool{" "}
          <code className="rounded bg-[#F4F1EA] px-1 font-mono text-[11px]">
            {String(d.tool)}
          </code>{" "}
          with query{" "}
          <span className="text-[#1C1C1E]">
            &ldquo;{String(d.query)}&rdquo;
          </span>{" "}
          in{" "}
          <span className="text-[#1C1C1E]">
            {Array.isArray(d.source_file)
              ? d.source_file.join(", ")
              : String(d.source_file)}
          </span>
          {d.model_wrote ? (
            <span className="text-[#9A968C]">
              {" "}
              (the model wrote &ldquo;{String(d.model_wrote)}&rdquo;, matched to
              the real file)
            </span>
          ) : null}
        </p>
      );
    case "context":
      return <ContextDetail detail={d} />;
    case "document":
      return <DocumentDetail detail={d} showPipeline={showPipeline} />;
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
  const firstDocument = trace.steps.findIndex((s) => s.kind === "document");

  return (
    <ChunksContext.Provider value={trace.chunks ?? {}}>
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
            ·{" "}
            {searches > 0
              ? `${searches} ${searches === 1 ? "search" : "searches"} · `
              : ""}
            {modelCalls} model {modelCalls === 1 ? "call" : "calls"} ·{" "}
            {formatMs(trace.total_ms)}
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
                      step.kind === "retrieval" ||
                      step.kind === "retrieval_docs" ||
                      step.kind === "agent"
                        ? "border-[#5E7C5C]"
                        : step.kind === "llm"
                          ? "border-[#C98AA2]"
                          : "border-[#CFC8B8]"
                    }`}
                    aria-hidden="true"
                  />
                  <div className="flex flex-wrap items-baseline justify-between gap-x-2">
                    <p>
                      <span className="mr-1.5 text-[10.5px] font-semibold text-[#9A968C]">
                        {KIND_LABEL[step.kind]}
                      </span>
                      <span className="font-medium text-[#1C1C1E]">
                        {step.title}
                      </span>
                    </p>
                    {step.ms !== undefined && (
                      <span className="tabular-nums text-[11px] text-[#9A968C]">
                        {formatMs(step.ms)}
                      </span>
                    )}
                  </div>
                  {step.kind === "retrieval" && (
                    <button
                      type="button"
                      onClick={() => setExpanded(expanded === i ? null : i)}
                      className="mt-0.5 text-[11px] font-medium text-[#5E7C5C] underline-offset-2 hover:underline"
                    >
                      {expanded === i
                        ? "Hide search results"
                        : "Show search results"}
                    </button>
                  )}
                  {isOpen && step.detail && (
                    <div className="mt-1">
                      <StepDetail
                        step={step}
                        showPipeline={i === firstDocument}
                      />
                    </div>
                  )}
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </ChunksContext.Provider>
  );
}
