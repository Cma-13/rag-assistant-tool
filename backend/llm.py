import json
import os
import re
import time
from dotenv import load_dotenv
from groq import RateLimitError, APIStatusError, APIConnectionError

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

import ollama as _ollama  # always available, used directly or as a fallback

_groq_client = None
if PROVIDER == "groq":
    from groq import Groq
    _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _call_ollama(prompt, num_predict, temperature):
    response = _ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"num_predict": num_predict, "temperature": temperature},
    )
    return response["response"]


def _call_groq(prompt, num_predict, temperature):
    kwargs = dict(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max(num_predict, 300),
        temperature=temperature,
    )
    if "gpt-oss" in GROQ_MODEL:
        kwargs["reasoning_effort"] = "low"
    response = _groq_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def generate(prompt, num_predict=200, temperature=0.1):
    """Send a prompt to the configured LLM provider. If Groq is active but
    rate-limited or unreachable, fall back to the local Ollama model for that
    one request, so the app keeps working instead of failing outright."""
    if PROVIDER == "groq":
        try:
            return _call_groq(prompt, num_predict, temperature)
        except (RateLimitError, APIStatusError, APIConnectionError) as e:
            print(f"Groq unavailable ({type(e).__name__}), falling back to Ollama for this request.")
            return _call_ollama(prompt, num_predict, temperature)
    else:
        return _call_ollama(prompt, num_predict, temperature)


def groq_chat(messages, tools=None, tool_choice="auto", temperature=0.1):
    if _groq_client is None:
        raise RuntimeError("groq_chat requires LLM_PROVIDER=groq")
    kwargs = dict(model=GROQ_MODEL, messages=messages, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if "gpt-oss" in GROQ_MODEL:
        kwargs["reasoning_effort"] = "low"
    return _groq_client.chat.completions.create(**kwargs)


def _parse_json(text):
    """Parse the model's reply as JSON. If it wrapped the JSON in extra text,
    pull out the part between the first { and the last }."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


class LLMBusyError(Exception):
    """Groq's usage limit is reached and waiting is not practical (for example
    the daily limit). Used instead of silently switching to the much weaker
    local model for jobs where a wrong answer is worse than no answer."""


def _retry_seconds(error, default=15.0, longest=45.0):
    """Groq's rate-limit message says 'try again in 12.5s' (or '1m3s'). Read how
    long to wait from it, never more than `longest` seconds."""
    m = re.search(r"try again in (?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", str(error))
    if m and (m.group(1) or m.group(2)):
        wait = int(m.group(1) or 0) * 60 + float(m.group(2) or 0)
        return wait + 1
    return default


def generate_json(prompt, schema, max_tokens=2000, temperature=0.0):
    """Ask the LLM for a JSON answer that matches `schema` and return it parsed.

    Groq: uses its structured-output mode (strict JSON schema), which forces
    the reply into exactly that shape instead of hoping the model formats it
    correctly. If Groq says "too many tokens this minute" (rate limit, error
    429) we wait the time it asks for and try again, up to 3 times, because the
    local Ollama model is much weaker at tables. If the limit is still there
    (or it is the daily limit) LLMBusyError is raised. Any other Groq failure
    falls back to Ollama's JSON mode for this request."""
    if PROVIDER == "groq":
        for attempt in range(3):
            try:
                kwargs = dict(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "structured_output", "strict": True, "schema": schema},
                    },
                )
                if "gpt-oss" in GROQ_MODEL:
                    kwargs["reasoning_effort"] = "low"
                response = _groq_client.chat.completions.create(**kwargs)
                return _parse_json(response.choices[0].message.content)
            except RateLimitError as e:
                wait = _retry_seconds(e)
                if wait > 60 or attempt == 2:
                    # A long wait means the daily limit is used up. The local
                    # model gives poor tables, so say "busy" instead.
                    print(f"Groq rate limit reached (would need to wait {wait:.0f}s), not using the local model for this.")
                    raise LLMBusyError(str(e))
                print(f"Groq rate limit (attempt {attempt + 1}/3), waiting {wait:.0f}s before retrying.")
                time.sleep(wait)
            except (APIStatusError, APIConnectionError, ValueError) as e:
                print(f"Groq structured output failed ({type(e).__name__}: {e}), falling back to Ollama for this request.")
                break

    response = _ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt + "\n\nRespond with JSON only.",
        format="json",
        options={"num_predict": max_tokens, "temperature": temperature},
    )
    return _parse_json(response["response"])