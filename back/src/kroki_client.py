# src/services/kroki_client.py
from __future__ import annotations
import requests
from typing import Literal, Tuple

KROKI_BASE = "https://kroki.io"

DiagramType = Literal["mermaid", "plantuml", "c4plantuml", "bpmn"]
OutFormat   = Literal["svg", "png"]

def render_kroki_sync(diagram_type: DiagramType, source: str, out: OutFormat = "svg") -> Tuple[bool, str | None, str | None]:
    """
    Renderiza usando la API HTTP de Kroki vía POST (más tolerante que el GET/deflate).
    Devuelve (ok, payload, error_msg). Para SVG el payload es una cadena XML.
    """
    url = f"{KROKI_BASE}/{diagram_type}/{out}"
    try:
        resp = requests.post(
            url,
            data=source.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=20,
        )
        if resp.status_code == 200:
            return True, resp.text, None
        else:
            return False, None, f"Kroki error {resp.status_code}: {resp.text[:400]}"
    except Exception as e:
        return False, None, f"Kroki exception: {e}"
