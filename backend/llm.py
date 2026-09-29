import os
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