import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if PROVIDER == "groq":
    from groq import Groq
    _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
else:
    import ollama as _ollama


def generate(prompt, num_predict=200, temperature=0.1):
    if PROVIDER == "groq":
        kwargs = dict(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            # gpt-oss models spend part of max_tokens on hidden reasoning before
            # producing visible text. A small budget (fine for Ollama) can be
            # entirely consumed by that, returning an empty string. Give more
            # headroom for Groq regardless of what was requested.
            max_tokens=max(num_predict, 300),
            temperature=temperature,
        )
        if "gpt-oss" in GROQ_MODEL:
            kwargs["reasoning_effort"] = "low"
        response = _groq_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    else:
        response = _ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"num_predict": num_predict, "temperature": temperature},
        )
        return response["response"]