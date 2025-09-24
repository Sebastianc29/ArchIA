# back/src/clients/kroki_client.py
from __future__ import annotations
import os
import time
import requests

# Puedes apuntar a tu instancia: http://<tu-vm>:8000  (si self-host)
KROKI_BASE = os.getenv("KROKI_BASE", "https://kroki.io")

ALLOWED_TYPES = {
    "plantuml", "c4plantuml", "graphviz", "mermaid",
    "erd", "vega", "vegalite", "svgbob", "structurizr"
}
ALLOWED_FORMATS = {"svg", "png", "pdf", "txt"}

def _normalize_type(diagram_type: str) -> str:
    t = (diagram_type or "").lower().strip()
    if t == "dot":
        t = "graphviz"
    return t

def render_kroki(diagram_type: str, source: str, output_format: str = "svg"):
    """
    Llamada base a Kroki: POST /{type}/{format} con text/plain.
    Devuelve (bytes, content_type) o lanza excepción en error.
    """
    t = _normalize_type(diagram_type)
    f = (output_format or "svg").lower().strip()

    if t not in ALLOWED_TYPES:
        raise ValueError(f"Tipo no soportado: {t}")
    if f not in ALLOWED_FORMATS:
        raise ValueError(f"Formato no soportado: {f}")

    url = f"{KROKI_BASE.rstrip('/')}/{t}/{f}"
    headers = {
        "Content-Type": "text/plain; charset=utf-8",
        "Accept": "image/svg+xml" if f == "svg" else "*/*",
    }

    # Normaliza saltos y quita fences si vienen del chat
    src = (source or "").replace("\r\n", "\n").strip()
    if src.startswith("```"):
        # elimina la primera línea de fence y deja el cuerpo
        parts = src.split("\n", 1)
        if len(parts) == 2:
            src = parts[1].strip("`").strip()

    backoff = 0.6
    for attempt in range(3):
        try:
            r = requests.post(url, data=src.encode("utf-8"), headers=headers, timeout=12)
            if r.status_code == 200:
                return r.content, r.headers.get("Content-Type", "image/svg+xml")
            if r.status_code in (413, 414):
                raise ValueError("Payload demasiado grande (413/414): usa POST, reduce tamaño o ajusta proxy.")
            if 500 <= r.status_code < 600:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise ValueError(f"Error Kroki {r.status_code}: {r.text[:200]}")
        except requests.Timeout:
            if attempt == 2:
                raise
    raise RuntimeError("Kroki no respondió tras reintentos.")

# --------- Wrappers compatibles con diagram_agent ---------

def render_kroki_sync(diagram_type: str, source: str, output_format: str = "svg", **kwargs):
    """
    Wrapper compatible con el código existente:
    - Acepta alias 'out' (p.ej., out='svg') usado por diagram_agent.
    - Devuelve (ok: bool, payload: bytes|None, err: str|None).
    """
    if "out" in kwargs and kwargs["out"]:
        output_format = kwargs["out"]
    try:
        content, _ctype = render_kroki(diagram_type, source, output_format)
        return True, content, None
    except Exception as e:
        return False, None, str(e)

# (Opcional) Async, por si algún nodo usa await
try:
    from starlette.concurrency import run_in_threadpool
    async def render_kroki_async(diagram_type: str, source: str, output_format: str = "svg", **kwargs):
        if "out" in kwargs and kwargs["out"]:
            output_format = kwargs["out"]
        try:
            content, _ctype = await run_in_threadpool(render_kroki, diagram_type, source, output_format)
            return True, content, None
        except Exception as e:
            return False, None, str(e)
except Exception:
    # Si no está starlette, simplemente no exponemos la variante async
    pass
