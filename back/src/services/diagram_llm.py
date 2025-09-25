from __future__ import annotations
import os, re, requests

PROVIDER = os.getenv("DIAGRAM_LLM_PROVIDER", "openai").lower()  # openai | ollama
# OpenAI (o compatibles)
OPENAI_BASE   = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_APIKEY = os.getenv("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """You are an expert software architect and PlantUML author.
Convert the user's natural-language request into a valid PlantUML DEPLOYMENT diagram.

HARD RULES:
- Output ONLY PlantUML between @startuml and @enduml (no prose, no code fences).
- ASCII only (no « », →, ↔, …). Use <<stereotypes>> and -> arrows.
- Prefer: cloud "Internet", node "Host" { nodes... }, database for DBs, folder for volumes.
- Show ports inside labels when provided.
- If relationships are missing, infer reasonable ones.
- Keep it compact and readable."""

_UNICODE_FIXES = [
    (r"```+plantuml|```+puml|```+", ""),
    (r"«", "<<"), (r"»", ">>"),
    (r"→|⇒|↦", "->"), (r"↔|⇄|⟷", "<->"),
    (r"—|–", "-"), (r"\u00A0", " ")
]

def _sanitize_puml(s: str) -> str:
    s = (s or "").strip()
    for pat, rep in _UNICODE_FIXES:
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    if "@startuml" not in s:
        s = "@startuml\n" + s
    if "@enduml" not in s:
        s = s + "\n@enduml"
    if "skinparam componentStyle" not in s:
        s = s.replace(
            "@startuml",
            "@startuml\nskinparam componentStyle uml2\nskinparam wrapWidth 200"
        )
    return s

def _call_openai(prompt: str) -> str:
    url = f"{OPENAI_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_APIKEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def generate_puml_from_nl(natural_prompt: str) -> str:
    raw = _call_openai(natural_prompt)
    return _sanitize_puml(raw)
