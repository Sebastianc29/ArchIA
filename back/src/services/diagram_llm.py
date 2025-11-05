from __future__ import annotations
import os, re
from langchain_core.messages import SystemMessage, HumanMessage
from src.services.llm_factory import get_chat_model

PROVIDER = os.getenv("DIAGRAM_LLM_PROVIDER") 
DIAGRAM_LLM_MODEL = os.getenv("DIAGRAM_LLM_MODEL")  

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

def _call_llm(prompt: str) -> str:
    llm = get_chat_model(provider=PROVIDER, model=DIAGRAM_LLM_MODEL, temperature=0.2, max_retries=2)
    msg = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
    return msg.content

def generate_puml_from_nl(natural_prompt: str) -> str:
    raw = _call_llm(natural_prompt)
    return _sanitize_puml(raw)
