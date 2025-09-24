from __future__ import annotations
import os
import requests
from typing import Tuple, Optional

PLANTUML_SERVER_URL = os.getenv("PLANTUML_SERVER_URL", "https://www.plantuml.com/plantuml")

def render_plantuml_sync(
    source: str,
    output_format: str = "svg",
    timeout: int = 20,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Renderiza PlantUML usando un PlantUML Server.
    - Prefiere POST /{format} con el texto PUML en el body (text/plain).
    - Devuelve (ok, payload_text, err). Para SVG, payload_text es str con XML.
    """
    fmt = (output_format or "svg").lower()
    if fmt not in {"svg", "png", "txt"}:
        fmt = "svg"

    url = PLANTUML_SERVER_URL.rstrip("/") + f"/{fmt}"
    headers = {"Content-Type": "text/plain; charset=utf-8"}
    try:
        resp = requests.post(url, data=source.encode("utf-8"), headers=headers, timeout=timeout)
        if resp.status_code == 200:
            # Para svg devolvemos texto, para png no (pero nuestra app usa svg)
            if fmt == "svg":
                return True, resp.text, None
            return True, resp.content, None  # por si algún día usas png
        return False, None, f"PlantUML HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, None, f"PlantUML error: {e}"