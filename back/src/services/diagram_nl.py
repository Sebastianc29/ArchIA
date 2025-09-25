from __future__ import annotations
import os, re, requests
from typing import Optional

# ====== Config proveedor LLM (por defecto OLLAMA local) ======
OLLAMA_BASE  = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

SYSTEM_NL_TO_PUML = """You are an expert software architect and PlantUML author.
Task: Convert the user's natural-language request into a **valid PlantUML DEPLOYMENT diagram**.
Rules:
- Output ONLY PlantUML code with @startuml ... @enduml (no explanations).
- Use ASCII only (avoid «», →, ↔, …). Use <<stereotypes>>, -> arrows, etc.
- Prefer nodes/databases/queues/folders for volumes. Add ports in node labels.
- If relationships are missing, infer the most reasonable ones (e.g., User->LB, LB->API, API->DB/Cache).
- Keep it compact (<= 4000 chars)."""

# Replace Unicode that rompe PlantUML
def _sanitize_puml(s: str) -> str:
    fixes = [
        (r"«", "<<"), (r"»", ">>"),
        (r"→|⇒|↦", "->"), (r"↔|⇄|⟷", "<->"),
        (r"—|–", "-"), (r"\u00A0", " "),
    ]
    for pat, rep in fixes:
        s = re.sub(pat, rep, s)
    return s

def _ensure_deployment_defaults(puml: str) -> str:
    # Si no definió @startuml/@enduml, envuelve
    if "@startuml" not in puml:
        puml = "@startuml\n" + puml
    if "@enduml" not in puml:
        puml = puml + "\n@enduml"
    # Añade pequeñas saneadas de estilo
    if "skinparam" not in puml:
        puml = puml.replace("@startuml", "@startuml\nskinparam componentStyle uml2\nskinparam wrapWidth 200")
    return puml

def _call_ollama(prompt: str) -> str:
    # Chat endpoint de Ollama
    url = f"{OLLAMA_BASE.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_NL_TO_PUML},
            {"role": "user",   "content": prompt},
        ],
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Respuesta típica: {"message":{"role":"assistant","content":"@startuml ... @enduml"}, "done":true}
    content = (data.get("message") or {}).get("content") or ""
    return content

def nl_to_plantuml(natural_prompt: str) -> str:
    """Recibe NL y devuelve código PlantUML DEPLOYMENT listo para Kroki."""
    # 1) Llama al LLM (Ollama por defecto)
    puml = _call_ollama(natural_prompt)
    # 2) Sanea unicode y garantiza @startuml/@enduml
    puml = _sanitize_puml(puml.strip())
    puml = _ensure_deployment_defaults(puml)

    # 3) Si no hay flechas, infiere relaciones estándar mínimas
    if "->" not in puml and "-->" not in puml:
        # Heurística básica: conecta Internet/User -> LB -> API, API -> DB/Cache/Queue si existen
        lines = [p for p in puml.splitlines()]
        has_lb   = any("Nginx" in l or "LB" in l or "loadbalancer" in l for l in lines)
        has_api  = any("API" in l for l in lines)
        has_db   = any("database" in l or "Postgres" in l or "DB" in l for l in lines)
        has_cache= any("Redis" in l or "<<cache>>" in l for l in lines)
        has_mq   = any("Rabbit" in l or "queue" in l for l in lines)

        edges = []
        if has_lb and has_api:
            edges += ['internet --> lb : HTTPS 443', 'lb --> api : HTTP 80']
        elif has_api:
            edges += ['internet --> api : HTTPS 443']
        if has_db:
            edges += ['api --> pg : SQL 5432']
        if has_cache:
            edges += ['api --> redis : TCP 6379']
        if has_mq:
            edges += ['api --> rabbit : AMQP 5672']

        if edges:
            # Inserta antes de @enduml
            puml = puml.replace("@enduml", "\n" + "\n".join(edges) + "\n@enduml")

    return puml
