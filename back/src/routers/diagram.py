# back/src/routers/diagram.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Tuple
import re

from ..clients.kroki_client import render_kroki

router = APIRouter(prefix="/diagram", tags=["diagram"])

# ---------------------- modelos ----------------------
class RenderReq(BaseModel):
    diagram_type: str
    source: str
    output_format: str = "svg"

class NLReq(BaseModel):
    prompt: str
    output_format: str = "svg"  # svg|png|pdf

# ---------------------- utilidades ----------------------
_UNICODE_FIXES = [
    (r"«", "<<"), (r"»", ">>"),
    (r"→|⇒|↦", "->"), (r"↔|⇄|⟷", "<->"),
    (r"—|–", "-"), (r"\u00A0", " "),
]

def _sanitize_plantuml(src: str) -> str:
    s = src
    for pat, rep in _UNICODE_FIXES:
        s = re.sub(pat, rep, s)
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

def _alias(name: str) -> str:
    # alias PlantUML seguro
    a = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip())
    a = re.sub(r"_+", "_", a).strip("_")
    if not a:
        a = "comp"
    return a.lower()

# estereotipo sugerido por nombre
def _guess_st(name: str) -> Tuple[str, str]:
    n = name.lower()
    # (shape, stereotype)
    if any(k in n for k in ["postgres", "mysql", "mariadb", "db", "database", "rds"]):
        return ("database", "db")
    if any(k in n for k in ["redis", "memcached", "cache"]):
        return ("node", "cache")
    if any(k in n for k in ["rabbit", "rabbitmq", "sqs", "queue", "broker"]):
        return ("node", "queue")
    if any(k in n for k in ["nginx", "ingress", "alb", "elb", "gateway", "load balancer", "loadbalancer", "lb"]):
        return ("node", "loadbalancer")
    if any(k in n for k in ["kafka"]):
        return ("node", "queue")
    if any(k in n for k in ["grafana", "prometheus", "tempo", "loki", "otel"]):
        return ("node", "monitoring")
    if any(k in n for k in ["api", "service", "svc"]):
        return ("node", "container")
    if any(k in n for k in ["web", "frontend", "react", "next", "angular", "vue"]):
        return ("node", "web")
    if any(k in n for k in ["worker", "celery", "consumer"]):
        return ("node", "container")
    return ("node", "")  # genérico

def _ports_from(text: str) -> str:
    # detecta "80/443", "8000", ":5432" en cualquier parte
    m = re.findall(r"(?<!\d)(\d{2,5}(?:/\d{2,5})*)(?!\d)", text)
    if not m:
        return ""
    return " :" + m[0]

def _extract_section(prompt: str, *keys: str) -> str | None:
    # Extrae texto después de "con:" / "componentes:" / etc. hasta fin de línea/ punto.
    p = prompt
    for k in keys:
        m = re.search(rf"{k}\s*:\s*([\s\S]+?)(?:$|[\n\.](?:\s|$))", p, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def _parse_components(prompt: str) -> List[Dict]:
    """
    Devuelve lista de componentes:
    [{"name": "Nginx LB", "alias":"nginx_lb", "label":"Nginx LB\\n:80/443", "shape":"node","st":"loadbalancer"}]
    """
    section = _extract_section(prompt, "con", "componentes", "components", "servicios")
    if not section:
        # fallback: intenta escanear todo el prompt (menos 'relaciones:' y 'volúmenes:')
        tmp = re.sub(r"(relaciones?|relations?)\s*:\s*[\s\S]+$", "", prompt, flags=re.IGNORECASE)
        section = re.sub(r"(vol[uú]menes?|volumes?)\s*:\s*[\s\S]+$", "", tmp, flags=re.IGNORECASE)

    parts = [x.strip() for x in re.split(r"[;,]", section) if x.strip()]
    comps: List[Dict] = []

    # Siempre meto "Internet" (entrada)
    comps.append({"name": "Internet", "alias": "internet", "label": "Internet", "shape": "cloud", "st": ""})

    for p in parts:
        # ejemplo: "API FastAPI 8000" -> name="API (FastAPI)", ports=" :8000"
        ports = _ports_from(p)
        name_raw = p
        if ports:
            name_raw = p.replace(ports.replace(" :", ""), "").strip()
        name = name_raw
        # compacta "API FastAPI" -> "API (FastAPI)" si son 2 tokens conocidos
        if "api" in name_raw.lower() and "fastapi" in name_raw.lower():
            name = "API (FastAPI)"
        shape, st = _guess_st(name)
        alias = _alias(name)
        label = name + (f"\\n{ports}" if ports else "")
        comps.append({"name": name, "alias": alias, "label": label, "shape": shape, "st": st})

    # El host para agrupar
    # Nota: no creo host si solo está Internet
    if len(comps) > 1:
        comps.insert(1, {"name": "Host", "alias": "host", "label": "Host", "shape": "node", "st": "_host_container"})

    return comps

def _parse_volumes(prompt: str) -> List[Tuple[str, str | None]]:
    """
    Devuelve listado de (vol_name, target_hint|None) sólo si el usuario los pide.
    Formatos aceptados: "Volúmenes: pgdata(Postgres), redisdata(Redis)"
    """
    text = _extract_section(prompt, r"vol[uú]menes", "volumes")
    if not text:
        return []
    out: List[Tuple[str, str | None]] = []
    for item in re.split(r"[;,]", text):
        t = item.strip()
        if not t:
            continue
        m = re.match(r"([A-Za-z0-9_\-\.]+)(?:\s*\(([^)]+)\))?", t)
        if not m:
            continue
        vname = m.group(1)
        target = m.group(2).strip() if m.group(2) else None
        out.append((vname, target))
    return out

def _parse_relations(prompt: str) -> List[str]:
    m = re.search(r"(relaciones?|relations?)\s*[:：]\s*([\s\S]+)$", prompt, re.IGNORECASE)
    if not m:
        return []
    raw = m.group(2)
    parts = re.split(r"[;\n,]+", raw)
    manual = [x.strip() for x in parts if ("->" in x or "-->" in x) and x.strip()]
    return manual

def _lookup_by_name(comps: List[Dict], name_hint: str) -> str | None:
    hint = name_hint.lower()
    best = None
    for c in comps:
        if c["alias"] in ("internet", "host"):
            continue
        n = c["name"].lower()
        if hint in n:
            best = c["alias"]
            break
    return best

def _build_edges_auto(comps: List[Dict]) -> List[str]:
    # heurística simple
    aliases = {c["alias"]: c for c in comps}
    have = lambda pred: [c for c in comps if pred(c)]
    lb = have(lambda c: c["st"] == "loadbalancer")
    web = have(lambda c: c["st"] == "web")
    api = have(lambda c: "api" in c["name"].lower() or c["st"] == "container")
    dbs = have(lambda c: c["st"] == "db")
    caches = have(lambda c: c["st"] == "cache")
    queues = have(lambda c: c["st"] == "queue")
    workers = have(lambda c: "worker" in c["name"].lower())

    edges: List[str] = []

    # Internet -> LB/Web/API
    if lb:
        edges.append(f'internet --> {lb[0]["alias"]} : HTTPS 443')
    elif web:
        edges.append(f'internet --> {web[0]["alias"]} : HTTPS 443')
    elif api:
        edges.append(f'internet --> {api[0]["alias"]} : HTTPS 443')

    # LB -> Web/API
    if lb and web:
        edges.append(f'{lb[0]["alias"]} --> {web[0]["alias"]} : HTTP 80')
    if lb and api:
        edges.append(f'{lb[0]["alias"]} --> {api[0]["alias"]} : HTTP 80')

    # Web -> API
    if web and api:
        edges.append(f'{web[0]["alias"]} --> {api[0]["alias"]} : HTTP')

    # API -> DB/Cache/Queue
    if api:
        a = api[0]["alias"]
        for d in dbs:
            edges.append(f'{a} --> {d["alias"]} : SQL 5432')
        for c in caches:
            edges.append(f'{a} --> {c["alias"]} : TCP 6379')
        for q in queues:
            edges.append(f'{a} --> {q["alias"]} : AMQP')

    # Worker -> Queue/Cache/DB
    for w in workers:
        if queues:
            edges.append(f'{w["alias"]} --> {queues[0]["alias"]} : consume')
        if caches:
            edges.append(f'{w["alias"]} --> {caches[0]["alias"]} : TCP')
        if dbs:
            edges.append(f'{w["alias"]} --> {dbs[0]["alias"]} : SQL')

    # elimina duplicados preservando orden
    seen = set()
    uniq = []
    for e in edges:
        if e not in seen:
            uniq.append(e)
            seen.add(e)
    return uniq

def _build_plantuml_deployment(prompt: str) -> str:
    comps = _parse_components(prompt)   # Internet + Host + cualquier cosa que hayas puesto
    vols = _parse_volumes(prompt)       # SOLO si el usuario los menciona
    manual_edges = _parse_relations(prompt)

    lines: List[str] = []
    lines.append("title Deployment (auto)")

    # Internet siempre fuera
    lines.append('cloud "Internet" as internet')

    # Host agrupa todo excepto Internet y Host mismo
    lines.append('node "Host" as host {')
    for c in comps:
        if c["alias"] in ("internet", "host"):
            continue
        if c["shape"] == "database":
            lines.append(f'  database "{c["label"]}" <<db>> as {c["alias"]}')
        else:
            st = f' <<{c["st"]}>>' if c["st"] else ""
            lines.append(f'  node "{c["label"]}"{st} as {c["alias"]}')
    lines.append("}")

    # volúmenes: solo si los pidió el usuario Y si logro mapearlos a un target
    vol_defs: List[str] = []
    vol_edges: List[str] = []
    for vname, target_hint in vols:
        target_alias = None
        if target_hint:
            target_alias = _lookup_by_name(comps, target_hint)
        else:
            # heurística: si nombre del volumen sugiere "pg" o "redis"
            if re.search(r"pg|postg", vname, re.IGNORECASE):
                target_alias = _lookup_by_name(comps, "postg")
            elif re.search(r"redis", vname, re.IGNORECASE):
                target_alias = _lookup_by_name(comps, "redis")
        if not target_alias:
            # si no encuentro target, NO dibujo este volumen
            continue
        vol_alias = _alias("vol_" + vname)
        vol_defs.append(f'folder "{vname}" as {vol_alias}')
        vol_edges.append(f'{target_alias} .. {vol_alias} : volume')

    # relaciones
    edges = manual_edges if manual_edges else _build_edges_auto(comps)

    all_lines = lines + vol_defs + edges + vol_edges
    return _sanitize_plantuml("\n".join(all_lines))

# ---------------------- endpoints ----------------------
@router.post("/render")
def render(req: RenderReq):
    dtype = req.diagram_type.strip().lower()
    src = req.source
    if dtype == "plantuml":
        src = _sanitize_plantuml(src)

    try:
        blob, ctype = render_kroki(dtype, src, req.output_format or "svg")
        return Response(content=blob, media_type=ctype)
    except Exception as e:
        # SOLO fallback para mermaid
        if dtype == "mermaid":
            return JSONResponse({"fallback": True, "syntax": "mermaid", "source": src})
        raise HTTPException(status_code=422, detail=f"Render failed for {dtype}: {e}")

@router.post("/nl")
def render_from_nl(req: NLReq):
    if not req.prompt or len(req.prompt.strip()) < 5:
        raise HTTPException(status_code=400, detail="Prompt vacío o demasiado corto.")

    puml = _build_plantuml_deployment(req.prompt)

    try:
        blob, ctype = render_kroki("plantuml", puml, req.output_format or "svg")
        return Response(content=blob, media_type=ctype)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Render failed: {e}")
