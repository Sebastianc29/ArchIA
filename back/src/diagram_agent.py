# src/diagram_agent.py
from __future__ import annotations
import base64, re
from typing import Optional, Tuple, List

from langchain_core.messages import AIMessage
from .clients.kroki_client import render_kroki_sync
from .clients.plantuml_local import render_plantuml_local

MAX_SOURCE_LEN = 20000

def _b64(s: str | bytes) -> str:
    if isinstance(s, str):
        s = s.encode("utf-8")
    return base64.b64encode(s).decode("ascii")

def _truncate(s: str, lim: int = MAX_SOURCE_LEN) -> str:
    return s if len(s) <= lim else (s[:lim] + "\n' [truncated]\n")

def _looks_like_component(prompt: str) -> bool:
    p = (prompt or "").lower()
    return ("component diagram" in p or "diagrama de componentes" in p or "uml2" in p) and ("deployment" not in p and "despliegue" not in p)

def _parse_field(prompt: str, key: str) -> Optional[str]:
    # Busca líneas tipo: key: valor
    m = re.search(rf"{re.escape(key)}\s*:\s*([^\n\r]+)", prompt, re.IGNORECASE)
    return m.group(1).strip() if m else None

def _parse_list(prompt: str, key: str) -> List[str]:
    raw = _parse_field(prompt, key) or ""
    # separa por coma; limpia comillas y espacios
    out = []
    for token in raw.split(","):
        t = token.strip().strip('"').strip("'")
        if t:
            out.append(t)
    return out

def _parse_relations(prompt: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Relaciones opcionales. Acepta formas como:
      StoreFront->OrderSystem
      StoreFront->Catalogue: ProductQuery
      OrderSystem->DB: Orders
    """
    text = _parse_field(prompt, "relaciones") or _parse_field(prompt, "relations") or ""
    rels = []
    for part in text.split():
        # permitimos separar por espacios si viene compactado; también aceptamos ';'
        for piece in part.split(";"):
            t = piece.strip()
            if not t:
                continue
            # busca "A->B(: etiqueta)?"
            m = re.match(r"([^-\s]+)\s*->\s*([^:\s]+)(\s*:\s*(.+))?$", t)
            if m:
                a = m.group(1).strip()
                b = m.group(2).strip()
                lab = (m.group(4) or "").strip() or None
                rels.append((a, b, lab))
    # además, intenta extraer por comas si venía en una sola línea
    if not rels and text:
        for piece in text.split(","):
            t = piece.strip()
            if not t:
                continue
            m = re.match(r"([^-\s]+)\s*->\s*([^:\s]+)(\s*:\s*(.+))?$", t)
            if m:
                a = m.group(1).strip()
                b = m.group(2).strip()
                lab = (m.group(4) or "").strip() or None
                rels.append((a, b, lab))
    return rels

def _build_component_puml(title: str, subsystem: str, comps: List[str], rels: List[Tuple[str,str,Optional[str]]]) -> str:
    """
    Construye un PlantUML de componentes. Si no hay relaciones, crea una topología razonable.
    """
    comps = [c for c in comps if c] or ["StoreFront", "OrderSystem", "PaymentGateway", "Auth", "Catalogue", "DB"]

    alias = {c: f"C_{re.sub(r'[^A-Za-z0-9_]', '_', c)}" for c in comps}

    # Wiring por defecto si no nos dieron relaciones
    if not rels:
        defaults = []
        if "StoreFront" in alias and "OrderSystem" in alias:
            defaults.append(("StoreFront", "OrderSystem", None))
        if "OrderSystem" in alias and "PaymentGateway" in alias:
            defaults.append(("OrderSystem", "PaymentGateway", "Payment"))
        if "OrderSystem" in alias and "DB" in alias:
            defaults.append(("OrderSystem", "DB", "SQL"))
        if "StoreFront" in alias and "Catalogue" in alias:
            defaults.append(("StoreFront", "Catalogue", "ProductQuery"))
        rels = defaults

    lines = [
        "@startuml",
        "skinparam backgroundColor white",
        "skinparam componentStyle rectangle",
        "skinparam wrapWidth 200",
        "skinparam maxMessageSize 80",
    ]
    if title:
        lines.append(f"title {title}")

    pkg_name = subsystem or "Subsystem"
    lines.append(f'package "{pkg_name}" <<Subsystem>> #E3F2FD {{')

    for c in comps:
        lines.append(f'  component "{c}" as {alias[c]}')
    lines.append("}")

    # Relación simple (flechas); si prefieres lollipops, puedes modelar interfaces explícitas
    for (a, b, lab) in rels:
        if a in alias and b in alias:
            if lab:
                lines.append(f'  {alias[a]} --> {alias[b]} : {lab}')
            else:
                lines.append(f'  {alias[a]} --> {alias[b]}')

    lines.append("@enduml")
    return "\n".join(lines) + "\n"

def _build_deployment_puml() -> str:
    # Tu mínimo para despliegue (o usa mermaid/plantuml según tu elección por Kroki)
    return (
        "@startuml\n"
        "skinparam backgroundColor white\n"
        "node \"User\" as U\n"
        "node \"Backend FastAPI\" as BE\n"
        "node \"Kroki Service\" as K\n"
        "U --> BE : HTTP(S)\n"
        "BE --> K : POST /{type}/{format}\n"
        "@enduml\n"
    )

def diagram_node(state: dict) -> dict:
    """
    Si detecto 'diagrama de componentes' => PlantUML local.
    Else => sigo usando Kroki (por ejemplo para despliegue).
    """
    user_q = state.get("userQuestion", "") or state.get("localQuestion", "") or ""
    is_component = _looks_like_component(user_q)

    if is_component:
        title = _parse_field(user_q, "título") or _parse_field(user_q, "titulo") or _parse_field(user_q, "title") or "Component Diagram"
        subsystem = _parse_field(user_q, "subsistema") or _parse_field(user_q, "subsystem") or "Subsystem"
        comps = _parse_list(user_q, "componentes") or _parse_list(user_q, "components")
        rels = _parse_relations(user_q)

        src = _build_component_puml(title, subsystem, comps, rels)
        src = _truncate(src)

        ok, payload, err = render_plantuml_local(src, out="svg")
        diagram = {
            "ok": bool(ok),
            "engine": "plantuml-local",
            "format": "svg",
            "svg_b64": _b64(payload) if (ok and isinstance(payload, str) and payload.strip().startswith("<svg")) else None,
            "data_uri": f"data:image/svg+xml;base64,{_b64(payload)}" if (ok and isinstance(payload, str) and payload.strip().startswith("<svg")) else None,
            "message": None if ok else (err or "PlantUML local error"),
            "source_echo": src,
        }

    else:
        # ejemplo: despliegue por Kroki (puedes dejar tu detección/branching real)
        src = _build_deployment_puml()
        ok, payload, err = render_kroki_sync("plantuml", src, "svg")
        diagram = {
            "ok": bool(ok),
            "engine": "kroki",
            "format": "svg",
            "svg_b64": _b64(payload) if (ok and payload) else None,
            "data_uri": f"data:image/svg+xml;base64,{_b64(payload)}" if (ok and payload) else None,
            "message": None if ok else (err or "Kroki error"),
            "source_echo": src,
        }

    state["diagram"] = diagram
    state["hasVisitedDiagram"] = True

    trace = (
        f"diagram_agent: ok={diagram['ok']} engine={diagram.get('engine')} "
        f"bytes={(len(diagram['svg_b64']) if diagram.get('svg_b64') else 0)}\n\n"
        f"--- SOURCE ---\n{diagram.get('source_echo','')}"
    )
    msgs = state.get("messages", [])
    msgs.append(AIMessage(name="diagram_agent", content=trace))
    state["messages"] = msgs
    return state
