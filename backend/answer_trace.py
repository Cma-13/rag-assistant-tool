"""Records how an answer was produced, step by step, so the app can show it.

Each question gets its own trace (a list of steps). Any part of the pipeline
can add a step with `add_step(...)`; when no trace is running (for example in
the MCP server) the calls do nothing. Recording never changes the answer."""

import contextvars
import time

_current = contextvars.ContextVar("docquery_trace", default=None)
_purpose = contextvars.ContextVar("docquery_trace_purpose", default=None)


def start():
    """Begin a new trace for this request."""
    trace = {"steps": [], "chunks": {}, "started": time.perf_counter()}
    _current.set(trace)
    return trace


def active():
    return _current.get() is not None


def add_step(kind, title, detail=None, ms=None):
    """kind: intent, rewrite, retrieval, agent, llm, context, document, note.
    detail: a small JSON-friendly dict with the numbers behind the step."""
    trace = _current.get()
    if trace is None:
        return
    step = {"kind": kind, "title": title}
    if detail:
        step["detail"] = detail
    if ms is not None:
        step["ms"] = round(ms)
    trace["steps"].append(step)


def add_step_before_last(before_kind, kind, title, detail=None):
    """Like add_step, but placed just before the most recent step of `before_kind`
    (used when we only learn something after that step has been recorded)."""
    trace = _current.get()
    if trace is None:
        return
    step = {"kind": kind, "title": title}
    if detail:
        step["detail"] = detail
    for i in range(len(trace["steps"]) - 1, -1, -1):
        if trace["steps"][i]["kind"] == before_kind:
            trace["steps"].insert(i, step)
            return
    trace["steps"].append(step)


def note_last_of(kind, **fields):
    """Add fields to the detail of the most recent step of this kind."""
    trace = _current.get()
    if trace:
        for step in reversed(trace["steps"]):
            if step["kind"] == kind:
                step.setdefault("detail", {}).update(fields)
                return


def note_last(**fields):
    """Add fields to the detail of the most recent step."""
    trace = _current.get()
    if trace and trace["steps"]:
        trace["steps"][-1].setdefault("detail", {}).update(fields)


def remember_chunk(chunk_id, info):
    """Keep one chunk's details (document, page, position, full text) once per trace;
    steps refer to it by id, so the same text is not repeated in every step."""
    trace = _current.get()
    if trace is not None and chunk_id is not None:
        trace["chunks"].setdefault(str(chunk_id), info)


def chunk(chunk_id):
    """The stored details of a chunk ({} when unknown)."""
    trace = _current.get()
    return trace["chunks"].get(str(chunk_id), {}) if trace else {}


def known_chunk(chunk_id):
    trace = _current.get()
    return trace is not None and str(chunk_id) in trace["chunks"]


def merge(other):
    """Add the steps and chunks another trace recorded (for example, the search steps
    the MCP server sends back with its result) to the current trace."""
    trace = _current.get()
    if trace is None or not other:
        return
    trace["steps"].extend(other.get("steps") or [])
    for cid, info in (other.get("chunks") or {}).items():
        trace["chunks"].setdefault(str(cid), info)


def step_count():
    trace = _current.get()
    return len(trace["steps"]) if trace else 0


def steps_since(index):
    """The steps added after position `index`."""
    trace = _current.get()
    return trace["steps"][index:] if trace else []


def first_step(kind):
    trace = _current.get()
    if trace:
        for step in trace["steps"]:
            if step["kind"] == kind:
                return step
    return None


def retitle_last(kind, title):
    """Give the most recent step of this kind a clearer title."""
    trace = _current.get()
    if trace:
        for step in reversed(trace["steps"]):
            if step["kind"] == kind:
                step["title"] = title
                return


class timer:
    """with timer() as t: ... ; then t.ms is the time taken in milliseconds."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.ms = 0
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._t0) * 1000
        return False


class purpose:
    """Names what the language model is being used for, so its calls are labelled
    in the trace. Use as `with purpose("Writing the answer"):` or as a decorator."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self._token = _purpose.set(self.label)
        return self

    def __exit__(self, *exc):
        _purpose.reset(self._token)
        return False

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            with purpose(self.label):
                return func(*args, **kwargs)
        wrapped.__name__ = func.__name__
        wrapped.__doc__ = func.__doc__
        return wrapped


def llm_call(provider, model, ms, fallback=False):
    """Record one language model call, labelled with the current purpose."""
    detail = {"provider": provider, "model": model}
    if fallback:
        detail["fallback"] = True
    add_step("llm", _purpose.get() or "Language model call", detail, ms)


def finish():
    """Stop the trace and return it in the shape the frontend shows."""
    trace = _current.get()
    _current.set(None)
    if trace is None:
        return None
    total = (time.perf_counter() - trace["started"]) * 1000
    return {"steps": trace["steps"], "chunks": trace["chunks"], "total_ms": round(total)}
