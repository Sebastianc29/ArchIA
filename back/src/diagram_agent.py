# src/diagram_agent.py
from __future__ import annotations
from typing import TypedDict
import base64, re

from langchain_core.messages import AIMessage
from .kroki_client import render_kroki_sync

# --------- helpers: construir un diagrama mínimo MERMAID ----------
def _build_min_mermaid_from_prompt(prompt: str) -> str:
    """
    Genera un diagrama de componentes mínimo y válido, basado en palabras clave.
    Preferimos mermaid (flowchart) porque es muy tolerante y Kroki lo soporta muy bien.
    """
    p = (prompt or "").lower()

    has_api  = any(k in p for k in ["api"])
    has_db   = any(k in p for k in ["db", "database", "datos"])
    has_auth = any(k in p for k in ["auth", "oauth", "jwt", "autentic"])

    lines = [
        "flowchart LR",
        "  client[Client]",
        "  api[API]",
        "  db[(Database)]",
        "  auth[[Auth Service]]",
        "",
        "  client --> api",
    ]
    if has_auth:
        lines += ["  api --> auth", "  auth --> api"]
    if has_db:
        lines += ["  api --> db", "  db --> api"]

    # por si no detectamos nada, deja al menos 3 nodos
    if not (has_api or has_db or has_auth):
        lines += ["  api --> db", "  db --> api"]

    return "\n".join(lines) + "\n"

def _svg_to_b64(svg_text: str) -> str:
    return base64.b64encode(svg_text.encode("utf-8")).decode("ascii")

# --------- LangGraph node ----------
def diagram_node(state: dict) -> dict:
    """
    Nodo sincrónico para LangGraph. Construye un MERMAID mínimo a partir del prompt,
    lo renderiza en Kroki y deja el resultado en state["diagram"] + una traza AIMessage.
    """
    user_q = state.get("userQuestion", "") or state.get("localQuestion", "")
    src_mermaid = _build_min_mermaid_from_prompt(user_q)

    ok, payload, err = render_kroki_sync("mermaid", src_mermaid, out="svg")
    diagram = {
        "ok": bool(ok),
        "format": "svg",
        "svg_b64": _svg_to_b64(payload) if (ok and payload) else None,
        "message": None if ok else (err or "Unknown Kroki error"),
        "source_echo": src_mermaid,
    }

    state["diagram"] = diagram
    state["hasVisitedDiagram"] = True

    # Deja una traza visible en LangSmith
    trace_text = (
        f"diagram_agent: ok={diagram['ok']} "
        f"bytes={len(diagram['svg_b64']) if diagram['svg_b64'] else 0}\n\n"
        f"--- MERMAID SOURCE ---\n{src_mermaid}"
    )
    state["messages"] = state.get("messages", []) + [AIMessage(name="diagram_agent", content=trace_text)]
    return state
