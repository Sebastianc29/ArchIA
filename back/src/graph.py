from __future__ import annotations
# ========== Imports

# Util
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pathlib import Path
import os, re, json, sqlite3, base64, logging
from dotenv import load_dotenv, find_dotenv
from src.utils.json_helpers import (
    extract_json_array,
    strip_first_json_fence,
    normalize_tactics_json,
    build_json_from_markdown,
)
# HTTP
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import json

# LangChain / LangGraph
from src.services.llm_factory import get_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# (Opcional) GCP Vision for image compare – protegido con try/except
try:
    from vertexai.generative_models import GenerativeModel
    from vertexai.preview.generative_models import Image
    _HAS_VERTEX = True
except Exception:
    _HAS_VERTEX = False

# RAG / citas
from src.rag_agent import get_retriever
# (estos pueden no usarse siempre, pero se conservan)
try:
    from src.quoting import pack_quotes, render_quotes_md
except Exception:
    pack_quotes = render_quotes_md = None

# ========== Token utils (soft) ==========
try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("gpt-4o")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text or ""))
except Exception:
    def _count_tokens(text: str) -> int:
        # aproximación si no hay tiktoken
        return max(1, int(len(text or "") / 3))

def _clip_text(text: str, max_tokens: int) -> str:
    if _count_tokens(text) <= max_tokens:
        return text
    target_chars = max(100, int(max_tokens * 3))  # 3 chars/token aprox
    return (text or "")[:target_chars] + "…"

def _clip_lines(lines: list[str], max_tokens: int) -> list[str]:
    out, total = [], 0
    for ln in lines:
        t = _count_tokens(ln)
        if total + t > max_tokens: break
        out.append(ln)
        total += t
    return out

def _last_k_messages(msgs, k=6):
    # Mantén solo los últimos K mensajes de usuario/asistente (sin repetir system)
    core = [m for m in msgs if getattr(m, "type", "") != "system"]
    return core[-k:]

# ========== Setup

load_dotenv(dotenv_path=find_dotenv('.env.development'))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("graph")
# === JSON parse hardening (tácticas) ===
import json, re

def _coerce_json_array(raw: str):
    """Intenta extraer un JSON array venga como venga (code-fence, texto suelto, etc.)."""
    if not raw:
        return None
    # 1) helper principal
    try:
        from src.utils.json_helpers import extract_json_array
        arr = extract_json_array(raw)
        if isinstance(arr, list) and arr:
            return arr
    except Exception:
        pass

    # 2) fence ```json ... ```
    m = re.search(r"```json\s*(\[.*?\])\s*```", raw, flags=re.I | re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 3) cualquier ``` ... ```
    m = re.search(r"```\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", raw, flags=re.I | re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, list) else None
        except Exception:
            pass

    # 4) primer array balanceado “best-effort”
    m = re.search(r"\[([\s\S]*)\]", raw)
    if m:
        txt = "[" + m.group(1) + "]"
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, list) else None
        except Exception:
            pass

    return None

# === Esquema y fallback por structured_output ===
TACTIC_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "purpose": {"type": "string"},
        "rationale": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "tradeoffs": {"type": "array", "items": {"type": "string"}},
        "categories": {"type": "array", "items": {"type": "string"}},
        "traces_to_asr": {"type": "string"},
        "expected_effect": {"type": "string"},
        "success_probability": {"type": "number"},
        "rank": {"type": "integer"}
    },
    "required": ["name","rationale","categories","success_probability","rank"]
}

TACTICS_ARRAY_SCHEMA = {
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "items": TACTIC_ITEM_SCHEMA
}

def _structured_tactics_fallback(llm, asr_text: str, qa: str, style_text: str):
    """Si el modelo no devolvió JSON, fuerza un array JSON válido de 3 tácticas."""
    prompt = f"""Return exactly THREE tactics that best satisfy this ASR.
Output ONLY JSON (no prose).

ASR:
{asr_text}

Primary quality attribute: {qa}
Selected style: {style_text or "(none)"}"""
    try:
        arr = llm.with_structured_output(TACTICS_ARRAY_SCHEMA).invoke(prompt)
        if isinstance(arr, list) and len(arr) == 3:
            return arr
    except Exception as e:
        log.warning("structured tactics fallback failed: %s", e)
    return None

BASE_DIR = Path(__file__).resolve().parent.parent  # back/
STATE_DIR = BASE_DIR / "state_db"
STATE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = STATE_DIR / "example.db"

conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
sqlite_saver = SqliteSaver(conn)

OPENAI_MODEL = os.getenv("ROS_LG_LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5-mini")
llm = get_chat_model(temperature=0.0)  # autodetecta por .env
retriever = get_retriever()

# ========== Diagrams backend endpoints / modos

# Nota: evita llamar al mismo puerto 8000 (mismo proceso) para no bloquear el bucle.
DIAGRAM_BASE = os.getenv("DIAGRAM_BASE", "http://127.0.0.1:8001").rstrip("/")
DIAGRAM_NL_MODE = os.getenv("DIAGRAM_NL_MODE", "fallback").lower()  # off | fallback | always
DIAGRAM_FORMAT = os.getenv("DIAGRAM_FORMAT", "svg")  # svg | png | pdf
DIAGRAM_INPROC = os.getenv("DIAGRAM_INPROC", "0") == "1"  # intenta uso en-proceso (si disponible)

# Sesión HTTP con retries y timeouts
def _make_http() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        connect=3,
        read=3,
        status=3,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET","POST"]),
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "ArchIA/diagram-orchestrator"})
    return s

_HTTP = _make_http()


# ========== Tatctics helpes
TACTICS_HEADINGS = [
    r"design tactics(?: to consider)?",
    r"tácticas(?: de diseño)?",
    r"arquitectural tactics",
    r"decisiones (?:arquitectónicas|de diseño)",
]

def _strip_tactics_sections(md: str) -> str:
    if not md:
        return md
    text = md
    for h in TACTICS_HEADINGS:
        text = re.sub(rf"(?is)\n+\s*{h}\s*:?.*$", "\n", text)
    return text.strip()

# --- Safe JSON example for tactics (avoid braces in f-strings) ---
TACTICS_JSON_EXAMPLE = """[
  {
    "name": "Elastic Horizontal Scaling",
    "purpose": "Keep p95 checkout latency under 200ms during 10x bursts",
    "rationale": "Autoscale replicas based on concurrency/CPU to avoid long queues violating the Response Measure",
    "risks": ["Higher peak spend", "Requires tuned HPA/policies"],
    "tradeoffs": ["Cost vs. resilience at peak"],
    "categories": ["scalability","latency","availability"],
    "traces_to_asr": "Stimulus=10x burst; Response=scale out; Response Measure=p95 < 200ms",
    "expected_effect": "Throughput increases and p95 stays under target during bursts",
    "success_probability": 0.82,
    "rank": 1
  }
]"""

# ========== PlantUML helper (LLM)

PLANTUML_SYSTEM = """
You are an expert software architect and PlantUML author.

The HUMAN message you receive is NOT just a short request: it is a multi-section prompt
with (approximately) this structure:

Business / project context:
<text>

Quality Attribute Scenario (ASR):
<full ASR in natural language>

Chosen architecture style:
<short style name>

Selected tactics:
- <tactic 1 name>
- <tactic 2 name>
- ...

User diagram request:
<short instruction, e.g. "Generate a deployment diagram aligned with these tactics.">

Your job is to TRANSFORM the ASR + style + tactics into a concrete architecture diagram,
NOT to draw the user sentence itself.

HARD RULES
- Output ONLY PlantUML between @startuml and @enduml (no prose, no fences).
- Use only ASCII (no arrows like →, etc.). Use <<stereotypes>> and -> arrows.
- Never create a component/node whose label is the text of the user request
  (e.g., "Generate a deployment diagram aligned with these tactics").
- The ASR artifact and response must drive the structure:
  - external client(s) / Internet
  - entrypoint (API gateway, web app, mobile app, etc.) that receives the stimulus
  - internal services/components that implement the tactics (e.g., cache, autoscaler,
    circuit breaker, message broker, DB, read replicas, CDN, etc.)
  - data stores, queues and monitoring components.

- If the user asks for a *deployment* diagram, model nodes (cloud/region, k8s cluster,
  hosts/VMs, databases, queues) and how components are deployed on them.
- If the user asks for a *component* diagram, focus on logical components and their
  connectors (no need to show physical nodes).

- Infer reasonable components and relationships from the ASR + style + tactics:
  - tie each tactic to at least one component or connector
    (e.g., "Elastic Horizontal Scaling" -> autoscaled API service,
           "Cache-Aside + TTL" -> cache in front of DB,
           "Circuit Breaker" -> proxy around downstream dependency).
  - make sure the diagram shows how the Response Measure in the ASR can be achieved
    (latency, throughput, availability, etc.).

- Prefer a compact but meaningful structure:
  - cloud "Internet" as entry.
  - node "k8s cluster" or "Cloud region" for infra.
  - inside, components/services, databases, queues, caches.

- Add arrows (->) between components to show data/control flow.
- Make the diagram readable and not overcrowded.
"""

MERMAID_SYSTEM = """
You are an expert software architect and Mermaid diagram author.

The HUMAN message you receive is NOT just a short request: it is a multi-section prompt
with (approximately) this structure:

Business / project context:
<text>

Quality Attribute Scenario (ASR):
<full ASR in natural language>

Chosen architecture style:
<short style name>

Selected tactics:
- <tactic 1 name>
- <tactic 2 name>
- ...

User diagram request:
<short instruction, e.g. "Generate a deployment diagram aligned with these tactics.">

Your job is to TRANSFORM the ASR + style + tactics into a concrete architecture diagram,
NOT to draw the user sentence itself.

IMPORTANT: The tactics and style you show in the diagram MUST come from the human prompt
(the ASR + style + selected tactics). Do NOT hard-code or always repeat the same tactics.
Use whatever tactics and style the upstream steps selected for this ASR.

===========================================================
HARD OUTPUT RULES (STRICT MERMAID SAFETY)
===========================================================

1. The FIRST line of output MUST ALWAYS be:
     graph LR
   Never place anything before it. Never omit it.

2. ALL Mermaid node IDs MUST match this regex:
     ^[a-zA-Z_][a-zA-Z0-9_]*$
   - No spaces, no hyphens, no dots, no trailing underscores.
   - Node IDs must be short and readable (api, cache, cb_proxy, edge_fn).

3. EVERY node MUST be declared BEFORE being used in an edge.
   Forbidden:
     api --> db["Database"]
   Required:
     db["Database"]
     api --> db

4. Node definitions MUST be on their own line.
   Forbidden inline definitions:
     api --> cb["Circuit Breaker"]
   Required:
     cb["Circuit Breaker"]
     api --> cb

5. ALL edges MUST follow EXACT Mermaid syntax:
     A --> B
     A --- B

   CRITICAL: In this system you MUST NOT use edge labels at all.
   That means:
     - Do NOT use: A --|label| B
     - Do NOT use: A -- text --> B
   Only unlabeled edges are allowed:
     - A --> B
     - A --- B

6. NO line may start with symbols or stray characters:
   Forbidden prefixes: "…", "|", ")", "}", "]", "_", "-", "•"
   Every line must begin with either:
     - nodeId
     - subgraph
     - end
     - whitespace + nodeId

7. ABSOLUTELY FORBIDDEN PATTERNS:
   - Inline nodes in edges
   - Two IDs glued together (e.g., clientcdn_cache, origin_inferenceedge_cb)
   - Incomplete IDs (edge_, cache__, _api)
   - Any label or ID that causes token merging
   - Edge labels with \\n or multi-line text
   - Unicode arrows or strange characters inside labels
   - Targeting quoted strings directly as edge endpoints
   - Creating tactic nodes inline in edges
   - Using reserved characters: `;`, `:`, `{}`, `[]` inside IDs

8. NEVER wrap the output in ``` fences.
   Output ONLY the Mermaid code.

===========================================================
SEMANTIC RULES (FROM YOUR ORIGINAL SYSTEM)
===========================================================

- The ASR artifact and response must drive the structure:
  external clients, entrypoints, internal services,
  caches, autoscaling, fallback, replication, DB, queues, monitoring.

- For deployment diagrams: use subgraphs for regions/clusters/hosts.
- For component diagrams: logical components only.

- Infer components from ASR + style + tactics.
- Tie tactics to components (cache, autoscaler, circuit breaker, etc.)

- Use short node IDs with readable labels:
     api["Checkout API"]
     cache["Redis Cache"]
     db[("Orders DB")]

- One Mermaid statement per line:
     node definition
     edge
     subgraph
     end

- Subgraphs MUST follow this pattern:
     subgraph REGION["Title"]
       node1["..."]
       node2["..."]
     end

===========================================================
TRACEABILITY / TACTICS (if applicable)
===========================================================

When you want to show which components implement which tactics,
use nodes for the tactics and connect components with unlabeled edges.

Example ONLY (you must adapt names to the REAL tactics from the prompt):
     tactic_cache["Tactic: Cache-Aside + TTL"]
     edge_cache --- tactic_cache
     precompute --- tactic_cache

     tactic_cb["Tactic: Circuit Breaker + Fallback"]
     cb_proxy --- tactic_cb

     tactic_scale["Tactic: Elastic Scaling"]
     autoscaler --- tactic_scale

These are just examples of structure. The actual tactic names and
number of tactics MUST come from the selected tactics in the human prompt.

===========================================================
EXTRA SAFETY RULES TO PREVENT MERMAID LEXICAL ERRORS
===========================================================

- Never produce labels containing slashes "/", commas ",", parentheses "( )",
  or long natural-language sentences. In fact, for this system you must NOT
  produce any edge labels at all; edges are plain arrows.

- Never let a line end with a node ID immediately followed by the next ID on
  the next line without a newline between them. This can cause Mermaid to merge
  IDs such as:
      cdn_edge
      edge_fn
  into the invalid token:
      cdn_edgeedge_fn

- To prevent this: the model SHOULD place a real blank line (an empty line with
  no spaces) between logically separate blocks (e.g., between different groups
  of edges or after subgraph blocks). However, a single newline between
  statements is still valid Mermaid. Do NOT put multiple statements on one line.

- Never place two edges or two node declarations on the same line.

- Do not generate ANY invisible characters, Unicode spaces, or hidden characters
  between IDs and arrows.

===========================================================
REMINDERS
===========================================================

- You are transforming ASR + style + tactics into a concrete architecture diagram.
- The chosen style and the selected tactics MUST be visible in the structure:
  components that embody those tactics, data paths that support the ASR metrics
  (latency, availability, throughput, degradation, etc.).
- Follow ALL rules above strictly so that the output parses correctly in Mermaid 11.x.

"""


def _sanitize_mermaid(code: str) -> str:
    if not code:
        return ""

    code = code.replace("\r\n", "\n")

    # Recortar cualquier texto antes del primer "graph" o "flowchart"
    m = re.search(r"(graph\s+(?:LR|TD|BT|RL)[\s\S]*$)", code, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"(flowchart[\s\S]*$)", code, flags=re.IGNORECASE)
    if m:
        code = m.group(1)
    else:
        # Si NO hay 'graph' ni 'flowchart', asumimos que son solo nodos/edges
        # y les anteponemos una cabecera por defecto.
        stripped = code.lstrip()
        if not stripped.lower().startswith(("graph ", "flowchart ")):
            code = "graph LR\n" + code

    # Reemplazar secuencias literales "\n" por espacios
    code = code.replace(r"\n", " ")

    # Normalizar algunos caracteres unicode problemáticos
    replacements = {
        "≤": "<=",
        "≥": ">=",
        "→": "->",
        "⇒": "->",
        "↔": "<->",
        "—": "-",
        "–": "-",
        "\u00A0": " ",
        "“": '"',
        "”": '"',
        "’": "'",
    }
    for bad, good in replacements.items():
        code = code.replace(bad, good)

    lines = code.split("\n")
    new_nodes: list[str] = []

    # 1) Patrones del tipo:  edge_cache ---|implements| "texto"
    edge_to_string = re.compile(
        r"^(\s*"               # indent + source id
        r"[A-Za-z_]\w*"        # id origen
        r"\s*)"
        r"(-{1,3}<?(?:>|)?)"   # operador de arista: --, -->, --- etc.
        r"\s*\|([^|]+)\|\s*"   # label
        r'"([^"]+)"\s*$'       # "texto" como destino
    )

    # 2) Patrones del tipo:  edge_cache --|MISS| cb["Circuit Breaker Proxy"]
    edge_with_inline_node = re.compile(
        r"^(\s*"               # indent + source id
        r"[A-Za-z_]\w*"
        r"\s*)"
        r"(-{1,3}<?(?:>|)?)"   # operador
        r"\s*\|([^|]+)\|\s*"   # label
        r"([A-Za-z_]\w*)\["    # id de nodo destino
        r"\"([^\"]+)\"\]\s*$"  # "texto" dentro del nodo
    )

    # Conjunto de nodos ya definidos (para no duplicar)
    defined_nodes = set()
    for line in lines:
        m_node = re.match(r"\s*([A-Za-z_]\w*)\s*\[", line)
        if m_node:
            defined_nodes.add(m_node.group(1))

    tactic_idx = 1

    for i, line in enumerate(lines):
        # Caso 1:  A ---|label| "texto"
        m1 = edge_to_string.match(line)
        if m1:
            indent, op, label, text = m1.groups()
            base_id = re.sub(r"\W+", "_", label.strip().lower()) or "note"
            node_id = f"tactic_{base_id}_{tactic_idx}"
            tactic_idx += 1

            # Definimos el nodo nuevo (nota/táctica)
            new_nodes.append(f'  {node_id}["{text}"]')
            defined_nodes.add(node_id)

            # Reemplazamos la línea original para que apunte al nodo
            lines[i] = f"{indent}{op} |{label.strip()}| {node_id}"
            continue

        # Caso 2:  A --|label| B["texto"]
        m2 = edge_with_inline_node.match(line)
        if m2:
            indent, op, label, node_id, text = m2.groups()

            # Reemplazamos la línea por edge hacia el id del nodo
            lines[i] = f"{indent}{op} |{label.strip()}| {node_id}"

            # Añadimos la definición del nodo si aún no existe
            if node_id not in defined_nodes:
                new_nodes.append(f'  {node_id}["{text}"]')
                defined_nodes.add(node_id)
            continue

    if new_nodes:
        lines.append("")
        lines.append("  %% Auto-generated tactic/note nodes")
        lines.extend(new_nodes)

    return "\n".join(lines).strip()


def _llm_nl_to_mermaid(natural_prompt: str) -> str:
    """
    Llama al LLM para obtener código Mermaid puro (sin fences) y lo sanea
    con _sanitize_mermaid antes de devolverlo.
    """
    msgs = [SystemMessage(content=MERMAID_SYSTEM),
            HumanMessage(content=natural_prompt)]
    resp = llm.invoke(msgs)
    raw = getattr(resp, "content", str(resp)) or ""

    # Si vino con ```mermaid ...```, usamos solo el cuerpo
    m = re.search(r"```mermaid\s*(.*?)```", raw, flags=re.I | re.S)
    if m:
        return _sanitize_mermaid(m.group(1))

    # Si vino con ```algo ...```, también usamos solo el cuerpo
    m = re.search(r"```(?:\w+)?\s*(.*?)```", raw, flags=re.I | re.S)
    if m:
        return _sanitize_mermaid(m.group(1))

    # Si no hay fences, saneamos todo el texto
    return _sanitize_mermaid(raw)

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

def _llm_nl_to_puml(natural_prompt: str) -> str:
    msgs = [SystemMessage(content=PLANTUML_SYSTEM),
            HumanMessage(content=natural_prompt)]
    resp = llm.invoke(msgs)
    raw = getattr(resp, "content", str(resp))
    return _sanitize_puml(raw)

# ========== Backends de render
def _render_nl_with_backend(uq: str, fmt: str = DIAGRAM_FORMAT) -> dict:
    """
    NL backend deshabilitado en esta instalación.
    Siempre devolvemos error para que el orquestador use LLM -> PlantUML -> Kroki.
    """
    return {"ok": False, "error": "NL backend disabled in this setup"}

def _render_puml_with_backend(puml: str, fmt: str = DIAGRAM_FORMAT) -> dict:
    """
    Renderiza PlantUML usando Kroki directamente, sin depender de un microservicio /diagram/render.
    """
    try:
        from src.clients.kroki_client import render_kroki_sync
        out_fmt = fmt or "svg"
        ok, payload, err = render_kroki_sync("plantuml", puml, out_fmt)
        if ok and payload:
            # payload ya viene en bytes desde Kroki
            b64 = base64.b64encode(payload).decode("ascii")
            return {
                "ok": True,
                "svg_b64": b64,
                "content_type": "image/svg+xml" if out_fmt == "svg" else "application/octet-stream",
            }
        return {"ok": False, "error": err or "Kroki render failed"}
    except Exception as e:
        return {"ok": False, "error": f"Kroki render error: {e}"}
def _render_inproc_puml(puml: str, fmt: str = DIAGRAM_FORMAT) -> dict:
    """Render en-proceso usando plantuml_local (binario local o JAR)."""
    if not DIAGRAM_INPROC:
        return {"ok": False, "error": "in-proc disabled"}
    try:
        from src.clients.plantuml_local import render_plantuml_local
        ok, payload, err = render_plantuml_local(puml, out=fmt or "svg")
        if not ok or not payload:
            return {"ok": False, "error": err or "PlantUML local error"}

        # Normalizamos a bytes para luego base64
        if isinstance(payload, str):
            svg_bytes = payload.encode("utf-8")
        else:
            svg_bytes = payload

        b64 = base64.b64encode(svg_bytes).decode("ascii")
        return {
            "ok": True,
            "svg_b64": b64,
            "content_type": "image/svg+xml" if (fmt or "svg") == "svg" else "application/octet-stream",
        }
    except Exception as e:
        return {"ok": False, "error": f"in-proc render error: {e}"}

# ========== Estado

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    userQuestion: str
    localQuestion: str
    hasVisitedInvestigator: bool
    hasVisitedCreator: bool
    hasVisitedEvaluator: bool
    hasVisitedASR: bool
    nextNode: Literal["investigator", "creator", "evaluator", "diagram_agent", "tactics", "asr", "style", "unifier"]
    doc_only: bool
    doc_context: str

    imagePath1: str
    imagePath2: str

    endMessage: str
    mermaidCode: str

    hasVisitedDiagram: bool
    diagram: dict

    # buffers / RAG / memoria liviana
    turn_messages: list
    retrieved_docs: list
    memory_text: str
    suggestions: list

    # memoria de conversación útil
    last_asr: str
    asr_sources_list: list

    # control de idioma/intención/forcing RAG
    language: Literal["en","es"]
    intent: Literal["general","greeting","smalltalk","architecture","diagram","asr","tactics","style"]
    force_rag: bool

    # etapa actual del pipeline ASR -> estilos -> tacticas -> despliegue
    arch_stage:str
    quality_attribute:str
    add_context:str
    tactics_list:list
    current_asr: str #ASR vigente
    tactics_md: str #salida markdonw del tactics_node
    tactics_struct: list #salida JSON parseada del tactics_node

    style: str #estilo actual
    selected_style: str
    last_style: str

class AgentState(TypedDict):
    messages: list
    userQuestion: str
    localQuestion: str
    imagePath1: str
    imagePath2: str


builder = StateGraph(GraphState)

# ========== Schemas

class supervisorResponse(TypedDict):
    localQuestion: Annotated[str, ..., "What is the question for the worker node?"]
    nextNode: Literal[
        "investigator",
        "creator",
        "evaluator",
        "diagram_agent",
        "tactics",
        "asr",
        "style",
        "unifier",
    ]

supervisorSchema = {
    "title": "SupervisorResponse",
    "description": "Response from the supervisor indicating the next node and the setup question.",
    "type": "object",
    "properties": {
        "localQuestion": {
            "type": "string",
            "description": "What is the question for the worker node?"
        },
        "nextNode": {
            "type": "string",
            "description": "The next node to act.",
            "enum": [
                "investigator",
                "creator",
                "evaluator",
                "unifier",
                "asr",
                "diagram_agent",
                "tactics",
                "style"          # ← AÑADIDO
            ]
        }
    },
    "required": ["localQuestion", "nextNode"]
}

class evaluatorResponse(TypedDict):
    positiveAspects: Annotated[str, ..., "What are the positive aspects of the user's idea?"]
    negativeAspects: Annotated[str, ..., "What are the negative aspects of the user's idea?"]
    suggestions: Annotated[str, ..., "What are the suggestions for improvement?"]

evaluatorSchema = {
    "title": "EvaluatorResponse",
    "description": "Response from the evaluator indicating the positive and negative aspects of the user's idea.",
    "type": "object",
    "properties": {
        "positiveAspects": {"type": "string"},
        "negativeAspects": {"type": "string"},
        "suggestions": {"type": "string"}
    },
    "required": ["positiveAspects", "negativeAspects", "suggestions"]
}

class investigatorResponse(TypedDict):
    definition: Annotated[str, ..., "What is the definition of the concept?"]
    useCases: Annotated[str, ..., "What are the use cases of the concept?"]
    examples: Annotated[str, ..., "What are the examples of the concept?"]

investigatorSchema = {
    "title": "InvestigatorResponse",
    "description": "Response from the investigator indicating the definition, use cases, and examples of the concept.",
    "type": "object",
    "properties": {
        "definition": {"type": "string"},
        "useCases": {"type": "string"},
        "examples": {"type": "string"}
    },
    "required": ["definition", "useCases", "examples"]
}

# ========== Heurísticas de idioma e intenciones

FOLLOWUP_PATTERNS = [
    ("explain_tactics", r"\b(tactics?|tácticas?).*(explain|describe|detalla|explica)|explica.*tácticas"),
    ("make_asr",        r"\b(asr|architecture significant requirement).*(make|create|example|ejemplo)|ejemplo.*asr"),
    ("component_view",  r"\b(component|diagrama de componentes|component diagram)"),
    ("deployment_view", r"\b(deployment|despliegue|deployment view)"),
    ("functional_view", r"\b(functional view|vista funcional)"),
    ("compare",         r"\b(compare|comparar).*?(latency|scalability|availability)"),
    ("checklist",       r"\b(checklist|lista de verificación|lista de verificacion)"),
]

def detect_lang(text: str) -> str:
    t = (text or "").lower()
    es_hits = sum(w in t for w in ["qué","que","cómo","como","por qué","porque","cuál","cual","hola","táctica","tactica","vista","despliegue"])
    en_hits = sum(w in t for w in ["what","how","why","which","hello","tactic","view","deployment","component"])
    if es_hits > en_hits: return "es"
    if en_hits > es_hits: return "en"
    return "en"

def classify_followup(question: str) -> str | None:
    q = (question or "").lower().strip()
    for intent, pat in FOLLOWUP_PATTERNS:
        if re.search(pat, q):
            return intent
    return None

# ========== ASR helpers

EVAL_TRIGGERS = [
    "evaluate this asr", "check this asr", "review this asr",
    "evalúa este asr", "evalua este asr", "revisa este asr",
    "es bueno este asr", "mejorar este asr", "mejorar asr",
    "critique this asr", "assess this asr"
]

def _looks_like_eval(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in EVAL_TRIGGERS)

def _wants_asr_evaluation(text: str) -> bool:
    t = (text or "").lower()
    if "asr" in t and any(k in t for k in EVAL_TRIGGERS):
        return True
    if any(k in t for k in ["evaluate", "check", "review", "evalúa", "evalua", "revisa"]) and "asr" in t:
        return True
    if "asr:" in t:
        return True
    return False

def _extract_candidate_asr(state: GraphState) -> str:
    """
    Prioridades:
    1) Si el usuario pegó `ASR: ...` en el mensaje, úsalo.
    2) Si no, usa el último ASR generado en este chat (state['last_asr'] / memory_text).
    3) En última instancia, usa el texto del usuario (por si es un ASR corto).
    """
    uq = state.get("userQuestion", "") or ""
    m = re.search(r"(?:^|\n)\s*ASR\s*:?(.*)$", uq, flags=re.I | re.S)
    if m and m.group(1).strip():
        return m.group(1).strip()

    q_lines = [ln.strip() for ln in uq.splitlines() if ln.strip()]
    if len(q_lines) == 1 and len(q_lines[0]) <= 200:
        return q_lines[0]

    last = state.get("last_asr") or ""
    if last:
        return last

    mem = state.get("memory_text", "") or ""
    mm = re.search(r"\[LAST_ASR\]\s*(.+)$", mem, flags=re.S)
    if mm:
        return mm.group(1).strip()

    return uq.strip()

# ========== Prompts base

def makeSupervisorPrompt(state: GraphState) -> str:
    visited_nodes = []
    if state["hasVisitedInvestigator"]: visited_nodes.append("investigator")
    if state["hasVisitedCreator"]:      visited_nodes.append("creator")
    if state["hasVisitedEvaluator"]:    visited_nodes.append("evaluator")
    if state.get("hasVisitedASR", False): visited_nodes.append("asr")
    visited_nodes_str = ", ".join(visited_nodes) if visited_nodes else "none"
    doc_flag = "ON" if state.get("doc_only") else "OFF"
    return f"""You are a supervisor orchestrating: investigator, creator (diagrams), evaluator, and ASR advisor.
Choose the next worker and craft a specific sub-question.

Rules:
- DOC-ONLY mode is {doc_flag}.
- If DOC-ONLY is ON: DO NOT call or suggest any retrieval tool (no local_RAG). Answers MUST rely only on the PROJECT DOCUMENT context provided.
- If DOC-ONLY is OFF and user asks about ADD/architecture, prefer investigator (and it may call local_RAG).
- If user asks for a diagram, route to creator.
- If user asks for an ASR or a QAS, route to asr.
- If two images are provided, evaluator may compare/analyze.
- Do not go directly to unifier unless at least one worker has produced output.

Visited so far: {visited_nodes_str}.
User question: {state["userQuestion"]}
Outputs: ['investigator','creator','evaluator','asr','unifier'].
"""

prompt_researcher = (
    "You are an expert in software architecture (ADD, quality attributes, tactics, views). "
    "When the question is architectural, you MUST call the tool `local_RAG` first, then optionally complement with LLM/LLMWithImages. "
    "Prefer verbatim tactic names from sources. Answer clearly and compactly.\n"
)

prompt_creator = "You are an expert in Mermaid and IT architecture. Generate a Mermaid diagram for the given prompt."

# ========== Helpers de turno

def _push_turn(state: GraphState, role: str, name: str, content: str) -> None:
    line = {"role": role, "name": name, "content": content}
    state["turn_messages"] = state.get("turn_messages", []) + [line]

def _reset_turn(state: GraphState) -> GraphState:
    return {**state, "turn_messages": []}

# ========== Tools (investigator / RAG / vision / evaluator) ==========
from langchain_core.tools import tool

@tool
def LLM(prompt: str) -> dict:
    """Researcher centrado en ADD/ADD 3.0.
    Devuelve un dict con [definition, useCases, examples] según investigatorSchema."""
    return llm.with_structured_output(investigatorSchema).invoke(prompt)

@tool
def LLMWithImages(image_path: str) -> str:
    """Analiza diagramas de arquitectura en imágenes (si Vertex AI está disponible)."""
    if not _HAS_VERTEX:
        return "Image analysis unavailable: Vertex AI SDK not installed."
    image = Image.load_from_file(image_path)
    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    resp = generative_multimodal_model.generate_content([
        ("Identify software-architecture tactics/patterns present. "
         "If the image is a class diagram, list classes/relations and OOD principles."),
        image
    ])
    return str(resp)

@tool
def local_RAG(prompt: str) -> str:
    """Responde con documentos locales (RAG) sobre tácticas/ADD/performance.
    Devuelve síntesis breve seguida de un bloque SOURCES para la UI."""
    q = (prompt or "").strip()
    synonyms = []
    if re.search(r"\badd\b", q, re.I):
        synonyms += ["Attribute-Driven Design", "ADD 3.0",
                     "architecture design method ADD", "Bass Clements Kazman ADD",
                     "quality attribute scenarios ADD"]
    if re.search(r"scalab|latenc|throughput|performance|tactic", q, re.I):
        synonyms += ["performance and scalability tactics", "latency tactics",
                     "scalability tactics", "architectural tactics performance"]

    queries = [q] + [f"{q} — {s}" for s in synonyms]
    docs_all = []
    for qq in queries:
        try:
            for d in retriever.invoke(qq):
                docs_all.append(d)
                if len(docs_all) >= 8:
                    break
        except Exception:
            pass
        if len(docs_all) >= 8:
            break

    # preview solo 2 y cada uno 400 chars
    preview = []
    for i, d in enumerate(docs_all[:2], 1):
        snip = (d.page_content or "").replace("\n", " ").strip()
        snip = (snip[:400] + "…") if len(snip) > 400 else snip
        preview.append(f"[{i}] {snip}")

    # fuentes máximo 6
    src_lines = []
    for d in docs_all[:6]:
        title = d.metadata.get("title") or Path(d.metadata.get("source_path", "")).stem or "doc"
        page = d.metadata.get("page_label") or d.metadata.get("page")
        src = d.metadata.get("source_path") or d.metadata.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        line = f"- {title}{page_str} — {src}"
        src_lines.append(_clip_text(line, 60))

    return "\n\n".join(preview) + "\n\nSOURCES:\n" + "\n".join(src_lines)

# ===== Evaluator tools =====
EVAL_THEORY_PREFIX = (
    "You are assessing the theoretical correctness of a proposed software architecture "
    "(patterns, tactics, views, styles). Be specific and concise."
)
EVAL_VIABILITY_PREFIX = (
    "You are assessing feasibility/viability (cost, complexity, operability, risks, team skill). "
    "Be realistic and actionable."
)
EVAL_NEEDS_PREFIX = (
    "You are checking alignment with user needs and architecture significant requirements (ASRs/QAS). "
    "Trace each point back to needs when possible."
)
ANALYZE_PREFIX = (
    "Compare two diagrams for the SAME component/system. Identify mismatches, missing elements and how they affect quality attributes."
)

@tool
def theory_tool(prompt: str) -> dict:
    """Evalúa corrección teórica vs buenas prácticas (patrones, tácticas, vistas)."""
    return llm.with_structured_output(evaluatorSchema).invoke(
        f"{EVAL_THEORY_PREFIX}\n\nUser input:\n{prompt}"
    )

@tool
def viability_tool(prompt: str) -> dict:
    """Evalúa viabilidad (coste, complejidad, operatividad, riesgos)."""
    return llm.with_structured_output(evaluatorSchema).invoke(
        f"{EVAL_VIABILITY_PREFIX}\n\nUser input:\n{prompt}"
    )

@tool
def needs_tool(prompt: str) -> dict:
    """Valida alineación con necesidades/ASRs y traza decisiones a requerimientos."""
    return llm.with_structured_output(evaluatorSchema).invoke(
        f"{EVAL_NEEDS_PREFIX}\n\nUser input:\n{prompt}"
    )

@tool
def analyze_tool(image_path: str, image_path2: str) -> str:
    """Compara dos diagramas de arquitectura (si Vertex AI está disponible)."""
    if not _HAS_VERTEX:
        return "Diagram compare unavailable: Vertex AI SDK not installed."
    image = Image.load_from_file(image_path)
    image2 = Image.load_from_file(image_path2)
    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    resp = generative_multimodal_model.generate_content([ANALYZE_PREFIX, image, image2])
    return str(resp)

# ========== Router

def router(state: GraphState) -> Literal["investigator","creator","evaluator","diagram_agent","tactics","asr","style","unifier"]:
    if state["nextNode"] == "unifier":
        return "unifier"

    # NEW: para peticiones de ASR con RAG, pasa primero por el investigador
    if state["nextNode"] == "asr" and not state.get("hasVisitedASR", False):
        if (
            not state.get("hasVisitedInvestigator", False)
            and not state.get("doc_only", False)
            and state.get("force_rag", False)
        ):
            return "investigator"
        return "asr"

    if state["nextNode"] == "style":
        return "style"
    elif state["nextNode"] == "tactics":
        return "tactics"
    elif state["nextNode"] == "investigator" and not state["hasVisitedInvestigator"]:
        return "investigator"
    elif state["nextNode"] == "creator" and not state["hasVisitedCreator"]:
        return "creator"
    elif state["nextNode"] == "diagram_agent" and not state.get("hasVisitedDiagram", False):
        return "diagram_agent"
    elif state["nextNode"] == "evaluator" and not state["hasVisitedEvaluator"]:
        return "evaluator"
    else:
        return "unifier"

# ========== Nodo: Orquestador de diagramas (NL → render / LLM→PUML→render) ==========
def diagram_orchestrator_node(state: GraphState) -> GraphState:
    """
    Nodo orquestador de diagramas:
    - Usa el ASR + estilo + tácticas + contexto + memoria del grafo
    - Genera SOLO el script Mermaid (state["mermaidCode"])
    - NO llama a Kroki, ni a /diagram/nl, ni genera SVG/PNG
    """
    # Pregunta actual del usuario (si existe)
    user_q = (state.get("localQuestion") or state.get("userQuestion") or "").strip()

    # --- ASR ---
    asr_text = (
        state.get("current_asr")
        or state.get("last_asr")
        or ""
    ).strip()

    # --- Estilo arquitectónico ---
    style_text = (
        state.get("style")
        or state.get("selected_style")
        or state.get("last_style")
        or ""
    ).strip()

    # --- Tácticas: preferimos la estructura JSON; si no, el markdown ---
    tactics_names: list[str] = []

    tactics_struct = state.get("tactics_struct") or []
    if isinstance(tactics_struct, list):
        for it in tactics_struct:
            if isinstance(it, dict) and it.get("name"):
                tactics_names.append(str(it["name"]))

    if not tactics_names:
        tactics_md = (state.get("tactics_md") or "").strip()
        if tactics_md:
            for line in tactics_md.splitlines():
                line = re.sub(r"^\s*[-*]\s*", "", line).strip()
                if line:
                    tactics_names.append(line)

    # Limitar un poco para que el diagrama no explote
    tactics_names = tactics_names[:8]

    tactics_block = (
        "\n".join(f"- {t}" for t in tactics_names)
        if tactics_names
        else "- (no explicit tactics selected yet)"
    )

    # --- Contexto / memoria adicional ---
    add_context = (state.get("add_context") or "").strip()
    doc_context = (state.get("doc_context") or "").strip()
    memory_text = (state.get("memory_text") or "").strip()

    sections: list[str] = []

    if add_context:
        sections.append(f"Business / project context:\n{add_context}")

    if doc_context:
        sections.append(f"Project documents context (RAG):\n{doc_context}")

    if memory_text:
        sections.append(f"Conversation memory (ASR/style/tactics decisions):\n{memory_text}")

    sections.append(
        "Quality Attribute Scenario (ASR):\n"
        f"{asr_text or '(not explicitly defined; infer it from the context and user request)'}"
    )

    sections.append(
        "Chosen architecture style:\n"
        f"{style_text or '(not explicitly chosen; infer a reasonable style for the ASR).'}"
    )

    sections.append("Selected tactics:\n" + tactics_block)

    sections.append(
        "User diagram request:\n"
        + (user_q or "Generate a deployment/component diagram aligned with the ASR and tactics.")
    )

    full_prompt = "\n\n---\n\n".join(sections)

    # --- Llamar al LLM especializado en Mermaid ---
    try:
        mermaid_code = _llm_nl_to_mermaid(full_prompt)
    except Exception as e:
        log.warning("diagram_orchestrator_node: Mermaid generation failed: %s", e)
        mermaid_code = ""

    state["mermaidCode"] = mermaid_code or ""
    # Ya no usamos imágenes ni backend de figuras
    state["diagram"] = {}
    state["hasVisitedDiagram"] = True
    # Opcional: marcar intención de diagrama por si algo más lo usa
    state["intent"] = "diagram"

    return state

# ========== Nodos principales ==========

def supervisor_node(state: GraphState):
    uq = (state.get("userQuestion") or "")

    # si ya hay un SVG listo en este turno, vamos directo al unifier
    d = state.get("diagram") or {}
    if d.get("ok") and d.get("svg_b64"):
        return {**state, "nextNode": "unifier", "intent": "diagram"}

    # idioma
    lang = detect_lang(uq)
    state_lang = "es" if lang == "es" else "en"

    # CORTE DE CIRCUITO: respeta la intención forzada desde main.py
    forced = state.get("intent")
    if forced == "asr":
        return {**state,
                "localQuestion": f"Create a concrete QAS (ASR) for: {uq}",
                "nextNode": "asr",
                "intent": "asr",
                "language": state_lang}
    
    if forced == "style":
        return {
            **state,
            "localQuestion": uq or (
                "Selecciona el estilo arquitectónico más adecuado para el ASR actual."
                if state_lang == "es"
                else "Select the most appropriate architecture style for the current ASR."
            ),
            "nextNode": "style",
            "intent": "style",
            "language": state_lang,
        }

    if forced == "tactics":
        return {**state,
                "localQuestion": ("Propose architecture tactics to satisfy the previous ASR. "
                                  "Explain why each tactic helps and ties to the ASR response/measure."),
                "nextNode": "tactics",
                "intent": "tactics",
                "language": state_lang}
    if forced == "diagram":
        return {**state,
                "localQuestion": uq,
                "nextNode": "diagram_agent",
                "intent": "diagram",
                "language": state_lang}

    # (a partir de aquí, SOLO si no vino intención forzada)
    fu_intent = classify_followup(uq)

        # --- NEW: detectar petición de estilos arquitectónicos ---
    style_terms = [
        "style", "styles",
        "architecture style", "architectural style",
        "estilo", "estilos", "estilo arquitectónico", "estilos arquitectónicos"
    ]
    wants_style = any(t in uq.lower() for t in style_terms)


    if _looks_like_eval(uq):
        return {**state,
                "localQuestion": uq,
                "nextNode": "evaluator",
                "intent": "architecture",
                "language": state_lang}

    # keywords para DIAGRAMAS (ES/EN)
    diagram_terms = [
        "diagrama", "diagrama de componentes", "diagrama de arquitectura",
        "diagram", "component diagram", "architecture diagram",
        "mermaid", "plantuml", "c4", "bpmn", "uml", "despliegue", "deployment"
    ]
    wants_diagram = any(t in uq.lower() for t in diagram_terms)

    # NEW: keywords para TÁCTICAS (ES/EN)
    tactics_terms = [
        "táctica", "tácticas", "tactic", "tactics",
        "estrategia", "estrategias", "strategy", "strategies",
        "cómo cumplir", "como cumplir", "how to satisfy",
        "how to meet", "how to achieve"
    ]
    wants_tactics = any(t in uq.lower() for t in tactics_terms)  # NEW

    sys_messages = [SystemMessage(content=makeSupervisorPrompt(state))]

    # baseline LLM (con fallback defensivo)
    try:
        resp = llm.with_structured_output(supervisorSchema).invoke(sys_messages)
        next_node = resp.get("nextNode", "investigator")
        local_q = resp.get("localQuestion", uq)
    except Exception:
        next_node, local_q = "investigator", uq

    intent_val = state.get("intent", "general")

        # 1) STYLE cuando el usuario lo pide explícitamente
    if wants_style or fu_intent == "style" or state.get("intent") == "style":
        next_node = "style"
        intent_val = "style"
        local_q = uq or (
            "Select the most appropriate architecture style for the current ASR."
            if state_lang == "en"
            else "Selecciona el estilo arquitectónico más adecuado para el ASR actual."
        )

    # 2) ASR después (no incluir tácticas aquí)
    elif any(x in uq.lower() for x in ["asr", "quality attribute scenario", "qas"]) or fu_intent == "make_asr":
        next_node = "asr"
        intent_val = "asr"
        local_q = f"Create a concrete QAS (ASR) for: {state['userQuestion']}"

    # 3) DIAGRAMA cuando lo piden
    elif wants_diagram or fu_intent in ("component_view", "deployment_view", "functional_view"):
        next_node = "diagram_agent"
        intent_val = "diagram"
        local_q = uq

    # 4) TÁCTICAS solo cuando el usuario las pide
    elif wants_tactics or fu_intent in ("explain_tactics", "tactics"):
        next_node = "tactics"
        intent_val = "tactics"
        local_q = (
            "Propose architecture tactics to satisfy the previous ASR. "
            "Explain why each tactic helps and how it ties to the ASR response/measure."
        )

    # 4) Resto
    elif fu_intent in ("compare", "checklist"):
        next_node = "investigator"
        intent_val = "architecture"

    # evita unifier si no se visitó nada este turno
    if next_node == "unifier" and not (
        state.get("hasVisitedInvestigator") or state.get("hasVisitedCreator") or
        state.get("hasVisitedEvaluator") or state.get("hasVisitedASR") or
        state.get("hasVisitedDiagram")
    ):
        next_node = "investigator"; intent_val = "architecture"

    return {
        **state,
        "localQuestion": local_q,
        "nextNode": next_node,
        "intent": intent_val,
        "language": state_lang
    }



def _sanitize_plain_text(txt: str) -> str:
    txt = re.sub(r"```.*?```", "", txt, flags=re.S)
    txt = txt.replace("**", "")
    txt = re.sub(r"^\s*#\s*", "", txt, flags=re.M)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def _dedupe_snippets(docs_list, max_items=3, max_chars=400) -> str:
    """Toma documentos del retriever y arma texto sin duplicados/ruido."""
    seen, out = set(), []
    for d in docs_list:
        t = (d.page_content or "").strip().replace("\n", " ")
        t = t[:max_chars]
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_items:
            break
    return "\n\n".join(out)

def asr_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    uq = state.get("userQuestion", "") or ""
    doc_only = bool(state.get("doc_only"))
    ctx_doc = (state.get("doc_context") or "").strip()

    # Heurística del atributo
    concern = "scalability" if re.search(r"scalab", uq, re.I) else \
              "latency"     if re.search(r"latenc", uq, re.I) else "performance"

    # Dominio típico si el usuario no lo da
    low = uq.lower()
    if any(k in low for k in ["e-comm", "commerce", "shop", "checkout"]):
        domain = "e-commerce flash sale"
    elif "api" in low:
        domain = "public REST API with burst traffic"
    elif any(k in low for k in ["stream", "kafka"]):
        domain = "event streaming pipeline"
    else:
        domain = "e-commerce flash sale"

    # === RAG (saltable) ===
    docs_list = []
    if state.get("force_rag", False) and not doc_only:
        try:
            query = f"{concern} quality attribute scenario latency measure stimulus environment artifact response response measure"
            docs_raw = list(retriever.invoke(query))
            docs_list = docs_raw[:6]
        except Exception:
            docs_list = []

    book_snippets = _dedupe_snippets(docs_list, max_items=6, max_chars=800)

    directive = "Answer in English." if lang == "en" else "Responde en español."
    ctx = (ctx_doc if (doc_only and ctx_doc) else (state.get("add_context") or "")).strip()[:2000]

    # 💡 NUEVO FORMATO: ASR complete + secciones planas
    prompt = f"""{directive}
You are an expert software architect following Attribute-Driven Design 3.0 (ADD 3.0).

Your job is to create ONE concrete Quality Attribute Scenario (Architecture Significant Requirement, ASR)
that will be used as an architectural driver.

The scenario MUST:
- Follow the classic QAS structure: Source, Stimulus, Environment, Artifact, Response, Response Measure.
- Be measurable, with a clear Response Measure (SLO/SLA, e.g. p95 < X ms under Y load, error rate, availability, etc.).
- Be realistic for production systems in the given domain.

Relevant domain or workload (you must stay coherent with this):
{domain}

Quality attribute focus inferred from the user message:
{concern}

User input to ground this ASR:
{uq}

PROJECT CONTEXT (if any):
{ctx or "None"}

OPTIONAL BOOK CONTEXT (only if not in DOC-ONLY mode):
{book_snippets or "None"}

OUTPUT FORMAT (MANDATORY – no bullets, no Markdown headings, no extra commentary):

ASR complete: <one single sentence that concisely states Source, Stimulus, Environment, Artifact, Response and Response Measure in natural language>

Scenario:
Source: <who initiates the stimulus>
Stimulus: <what happens / event that triggers the behavior>
Environment: <when / in which operating conditions this happens>
Artifact: <what part of the system is stimulated>
Response: <what the system must do>
Response Measure: <how success is measured with clear numeric thresholds>

Rules:
- The line that starts with "ASR complete:" MUST be a single sentence.
- Then a blank line.
- Then the section "Scenario:" in its own line and each of the six fields (Source, Stimulus, Environment, Artifact, Response, Response Measure)
  on its own line exactly as shown above.
- Do NOT add any other sections (no 'Architectural Driver Summary', no 'Summary', no 'Context' headings).
- Do NOT talk about tactics, styles or next steps here.
- Keep the numbers realistic and monitorable (p95 / p99, RPS, error rate, availability, etc.).
- Answer entirely in the requested language.
"""

    result = llm.invoke(prompt)
    content_raw = getattr(result, "content", str(result))
    content = _sanitize_plain_text(content_raw)
    content = _strip_tactics_sections(content)

    # === Fuentes (si hubo RAG) ===
    src_lines = []
    for d in (docs_list or []):
        md = d.metadata or {}
        title = md.get("source_title") or md.get("title") or "doc"
        page  = md.get("page_label") or md.get("page")
        path  = md.get("source_path") or md.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        src_lines.append(f"- {title}{page_str} — {path}")
    if src_lines:
        src_lines = [_clip_text(s, 60) for s in src_lines]
        src_lines = list(dict.fromkeys(src_lines))[:4]

    src_block = "SOURCES:\n" + ("\n".join(src_lines) if src_lines else "- (no local sources)")

    # Traza + memoria de turno
    state["turn_messages"] = state.get("turn_messages", []) + [
        {"role": "system", "name": "asr_system", "content": prompt},
        {"role": "assistant", "name": "asr_recommender", "content": content},
        {"role": "assistant", "name": "asr_sources", "content": src_block},
    ]
    state["messages"] = state["messages"] + [
        AIMessage(content=content, name="asr_recommender"),
        AIMessage(content=src_block, name="asr_sources"),
    ]

    # Memoria viva del chat
    state["last_asr"] = content
    refs_list = [ln.lstrip("- ").strip() for ln in src_block.splitlines()
                 if ln.strip() and not ln.lower().startswith("sources")]
    state["asr_sources_list"] = refs_list
    prev_mem = state.get("memory_text", "") or ""
    state["memory_text"] = (prev_mem + f"\n\n[LAST_ASR]\n{content}\n").strip()

    # Metadatos
    state["quality_attribute"] = concern
    state["arch_stage"] = "ASR"
    state["current_asr"] = content

    # Señales de fin de turno
    state["endMessage"] = content
    state["hasVisitedASR"] = True
    state["force_rag"] = False
    state["nextNode"] = "unifier"

    return state

def style_node(state: GraphState) -> GraphState:
    """
    Architecture style node (ADD 3.0):

    - Proposes EXACTLY 2 candidate styles.
    - Evaluates the impact of each one on the ASR.
    - Recommends one of them.
    - Stores only the recommended style as the active style in the pipeline.
    """
    lang = state.get("language", "es")
    directive = "Answer in English." if lang == "en" else "Responde en español."

    # 1) Recover ASR, quality attribute, and business context
    asr_text = (
        state.get("current_asr")
        or state.get("last_asr")
        or (state.get("userQuestion") or "")
    )
    qa = state.get("quality_attribute", "")
    ctx = (state.get("add_context") or "").strip()

    # 2) Prompt: ask for JSON with 2 styles + recommendation (PROMPT 100% IN ENGLISH)
    prompt = f"""{directive}
You are a software architect applying ADD 3.0.

Given the following Quality Attribute Scenario (ASR) and its business context,
propose exactly TWO different architecture styles as reasonable candidates,
and then recommend which of the two is BETTER to satisfy this ASR,
explaining the recommendation in terms of its impact on the quality attribute.

Quality attribute focus (e.g., availability, performance, latency, security, etc.):
{qa}

Business / context:
{ctx or "(none)"}

ASR:
{asr_text}

You MUST respond with a VALID JSON object ONLY, with NO extra text, in the following form:

{{
  "style_1": {{
    "name": "Short name of style 1 (e.g., 'Layered', 'Microservices')",
    "impact": "Brief description of how this style impacts the ASR (pros, cons, trade-offs)."
  }},
  "style_2": {{
    "name": "Short name of style 2",
    "impact": "Brief description of how this style impacts the ASR (pros, cons, trade-offs)."
  }},
  "best_style": "style_1 or style_2 (choose ONE)",
  "rationale": "Explain why the chosen style is better for this ASR, based on its impact."
}}

Do NOT add comments or any text outside of this JSON object.
"""

    result = llm.invoke(prompt)
    raw = getattr(result, "content", str(result))

    # 3) Parse JSON (fallback if it fails)
    try:
        data = json.loads(raw)
    except Exception:
        # If no valid JSON: at least store one line as style
        fallback_style = raw.splitlines()[0].strip()
        state["style"] = fallback_style
        state["selected_style"] = fallback_style
        state["last_style"] = fallback_style
        state["arch_stage"] = "STYLE"
        state["endMessage"] = raw
        state["nextNode"] = "unifier"
        return state

    style1 = data.get("style_1", {}) or {}
    style2 = data.get("style_2", {}) or {}
    style1_name = style1.get("name", "").strip() or "Style 1"
    style2_name = style2.get("name", "").strip() or "Style 2"
    style1_impact = style1.get("impact", "").strip()
    style2_impact = style2.get("impact", "").strip()
    best_key = (data.get("best_style") or "").strip()
    rationale = data.get("rationale", "").strip()

    if best_key == "style_2":
        chosen_name = style2_name
    else:
        # Default to style_1 if not clear
        best_key = "style_1"
        chosen_name = style1_name

    # 4) Store ONLY the recommended style in the ADD 3.0 state
    state["style"] = chosen_name
    state["selected_style"] = chosen_name
    state["last_style"] = chosen_name
    state["arch_stage"] = "STYLE"

    # Update rich memory (long-term text)
    prev_mem = state.get("memory_text", "") or ""
    state["memory_text"] = (
        prev_mem
        + f"\n\n[STYLE_OPTIONS]\n1) {style1_name}\n2) {style2_name}\n"
        + f"[STYLE_CHOSEN]\n{chosen_name}\n"
    ).strip()

    # 5) Build user-facing message (in ES or EN)
    if lang == "es":
        header = "He identificado dos estilos arquitectónicos candidatos para tu ASR:"
        rec_label = "Recomendación"
        because = "porque"
        followups = [
            f"Explícame tácticas concretas para el ASR usando el estilo recomendado ({chosen_name}).",
            "Compárame más a fondo estos dos estilos para este ASR.",
        ]
    else:
        header = "I have identified two candidate architecture styles for your ASR:"
        rec_label = "Recommendation"
        because = "because"
        followups = [
            f"Explain concrete tactics for the ASR using the recommended style ({chosen_name}).",
            "Compare these two styles in more depth for this ASR.",
            ]

    content = (
        f"{header}\n\n"
        f"1) {style1_name}\n"
        f"   - Impact: {style1_impact}\n\n"
        f"2) {style2_name}\n"
        f"   - Impact: {style2_impact}\n\n"
        f"{rec_label}: **{chosen_name}** {because}:\n"
        f"{rationale}\n"
    )

    state["turn_messages"] = state.get("turn_messages", []) + [
        {"role": "assistant", "name": "style_recommender", "content": content}
    ]
    state["suggestions"] = followups
    state["endMessage"] = content
    state["nextNode"] = "unifier"

    return state



def _guess_quality_attribute(text: str) -> str:
    low = (text or "").lower()
    if "latenc" in low or "response time" in low: return "latency"
    if "scalab" in low or "throughput" in low:    return "scalability"
    if "availab" in low or "uptime" in low:       return "availability"
    if "secur" in low:                             return "security"
    if "modifiab" in low or "change" in low:       return "modifiability"
    if "reliab" in low or "fault" in low:          return "reliability"
    return "performance"
def _json_only_repair_pass(llm, asr_text: str, qa: str, style_text: str, md_preview: str) -> list | None:
    prompt = f"""You previously wrote tactics in markdown but missed the JSON.
Return ONLY ONE code fence with a JSON array of EXACTLY 3 objects. No prose.

Schema example (do not copy values, just shape):
[
  {{
    "name": "Tactic name",
    "purpose": "Short purpose",
    "rationale": "Why it satisfies the ASR",
    "risks": ["..."],
    "tradeoffs": ["..."],
    "categories": ["scalability","latency"],
    "traces_to_asr": "Stimulus=..., Response=..., Response Measure=...",
    "expected_effect": "Short",
    "success_probability": 0.8,
    "rank": 1
  }}
]

ASR:
{asr_text}

Primary quality attribute: {qa}
Selected style: {style_text or "(none)"}

Previous markdown (for grounding):
{md_preview[:1800]}
"""
    resp = llm.invoke(prompt)
    raw = getattr(resp, "content", str(resp))
    struct = extract_json_array(raw)
    return struct if (isinstance(struct, list) and len(struct) >= 1) else None

def tactics_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    directive = "Answer in English." if lang == "en" else "Responde en español."
    doc_only = bool(state.get("doc_only"))
    ctx_doc = (state.get("doc_context") or "").strip()
    ctx_add = (state.get("add_context") or "").strip()
    ctx = (ctx_doc if (doc_only and ctx_doc) else ctx_add)[:2000]

    # 1) Tomamos el ASR actual (o lo inferimos del mensaje)
    asr_text = state.get("asr_text") or state.get("last_asr") or ""
    if not asr_text:
        uq = state.get("userQuestion", "") or ""
        m = re.search(r"(?:^|\n)\s*ASR\s*:\s*(.+)$", uq, flags=re.I | re.S)
        asr_text = (m.group(1).strip() if m else uq.strip())

    # 2) Deducimos el atributo de calidad
    qa = state.get("quality_attribute") or _guess_quality_attribute(asr_text)
    # Estilo (si lo trae el flujo de ESTILOS)
    style_text = state.get("style") or state.get("selected_style") or state.get("last_style") or ""

    # 3) Contexto para grounding: DOC-ONLY → sin RAG; otro caso → RAG normal
    docs_list = []
    if doc_only and ctx_doc:
        book_snippets = f"[DOC] {ctx_doc[:2000]}"
    else:
        try:
            queries = [
                f"{qa} architectural tactics",
                f"{qa} tactics performance scalability latency availability security modifiability",
                "Bass Clements Kazman performance and scalability tactics",
                "quality attribute tactics list"
            ]
            seen = set()
            gathered = []
            for q in queries:
                for d in retriever.invoke(q):
                    key = (d.metadata.get("source_path"), d.metadata.get("page"))
                    if key in seen:
                        continue
                    seen.add(key)
                    gathered.append(d)
                    if len(gathered) >= 6:
                        break
                if len(gathered) >= 6:
                    break
            docs_list = gathered
        except Exception:
            docs_list = []
        book_snippets = _dedupe_snippets(docs_list, max_items=5, max_chars=600)
    JSON_EXAMPLE = TACTICS_JSON_EXAMPLE
    # 4) Prompt: pedimos Markdown + JSON
    prompt = f"""{directive}
You are an expert software architect applying Attribute-Driven Design 3.0 (ADD 3.0).

We ALREADY HAVE an ASR / Quality Attribute Scenario. That ASR is an ADD 3.0 architectural driver.
Your job now is to continue the ADD 3.0 process by selecting architectural tactics.

PROJECT CONTEXT (if any)
{ctx or "None"}

ASR (driver to satisfy):
{asr_text or "(none provided)"}

Primary quality attribute (guessed):
{qa}
Selected architecture style (if any):
{style_text or "(none)"}


GROUNDING (use ONLY this context; if DOC-ONLY, this is the exclusive source):
{book_snippets or "(none)"}

If DOC-ONLY is ON, do not rely on knowledge beyond the PROJECT DOCUMENT even if you “know” typical tactics. If the document does not support a tactic, state “not supported by the document”.

You MUST output THREE sections, in EXACT order:

(0) Which is the ASR and it´s style (if any):
- 3–5 concise lines.
- Explicitly link back to the ASR's Source, Stimulus, Artifact, Environment and Response Measure. Also its architectonic style. Example: "The external clients and bots, when a 10x traffic burst during product drop, (checkout API) in a normal operation and one region, the system must keep throughput and protect downstreams with a response measure of p95 < 200ms and error rate < 0.5% using microservices with API gateway and Redis cache."

(1) TACTICS (TOP-3 with highest success probability):
Select EXACTLY THREE architectural tactics that maximally satisfy this ASR GIVEN the selected style.
For EACH tactic include:
- Name — canonical tactic name (e.g., "Elastic Horizontal Scaling", "Cache-Aside + TTL", "Circuit Breaker").
- Rationale — why THIS tactic directly satisfies THIS ASR's Response & Response Measure in THIS style.
- Consequences / Trade-offs — realistic costs/risks (cost, complexity, ops burden, coupling, failure modes).
- When to use — explicit runtime trigger/guard (e.g., "if p95 > 200ms during 10x burst for 1 minute, trigger X").
- Why it ranks in TOP-3 — short argument grounded on ASR + style fit.
- Sucess probability — numeric estimate [0,1] of success in production.

(2) JSON:
Return ONE code fence starting with ```json and ending with ``` that contains ONLY a JSON array with EXACTLY 3 objects.
- Use dot as decimal separator (e.g., 0.82), never commas.
- Do not use percent signs, just 0..1 floats for success_probability.
- Do not add any prose or markdown outside the JSON fence.

Example shape (values are illustrative — adjust to your tactics):
{JSON_EXAMPLE}


STRICT RULES:
- You MUST behave like ADD 3.0: tactics are chosen BECAUSE OF the ASR's Response and Response Measure, not randomly.
- Every tactic MUST explicitly tie back to the ASR driver.
- DO NOT invent product names or vendor SKUs. Stay pattern-level.
- Keep output concise, production-realistic, and auditable.
- Output EXACTLY 3 tactics — do not list more than 3.
- Provide a numeric "success_probability" in [0,1] and a unique "rank" (1..3) consistent with the markdown ranking.
"""
    resp = llm.invoke(prompt)
    raw = getattr(resp, "content", str(resp)).strip()

    # LOG opcional (útil para depurar)
    log.debug("tactics raw (first 400): %s", raw[:400].replace("\n"," "))
    log.debug("has ```json fence? %s", bool(re.search(r"```json", raw, re.I)))

    # 5) Parseo + reparación en cascada (solo helpers existentes)
    struct = extract_json_array(raw) or []

    if not (isinstance(struct, list) and struct):
        struct = _json_only_repair_pass(
            llm, asr_text=asr_text, qa=qa, style_text=style_text, md_preview=raw
        ) or []

    if not (isinstance(struct, list) and struct):
        struct = build_json_from_markdown(raw, top_n=3)

    # Normaliza a TOP-3 + shape final
    struct = normalize_tactics_json(struct, top_n=3)
    log.info(
        "tactics_struct.len=%s names=%s",
        len(struct) if isinstance(struct, list) else 0,
        [it.get("name") for it in (struct or []) if isinstance(it, dict)]
    )

        
    # Markdown a mostrar (remueve el primer bloque ```json del modelo si vino)
    md_only = strip_first_json_fence(raw)

    show_json = os.getenv("SHOW_TACTICS_JSON", "0") == "1"
    if show_json:
        md_only = f"{md_only}\n\n```json\n{json.dumps(struct, ensure_ascii=False, indent=2)}\n```"
    else:
        # si ocultas el JSON, borra el encabezado "(2) JSON:" que queda colgando
        md_only = re.sub(r"\n?\(?2\)?\s*JSON\s*:?\s*$", "", md_only, flags=re.I|re.M).rstrip()

    # Fallback visual si por alguna razón no hay markdown
    if (not md_only) and isinstance(struct, list) and struct:
        md_only = "\n".join(
            f"- {it.get('name','')}: {it.get('rationale','')}"
            for it in struct if isinstance(it, dict)
        )

    # 6) Fuentes
    src_lines = []
    for d in (docs_list or []):
        md = d.metadata or {}
        title = md.get("source_title") or md.get("title") or "doc"
        page  = md.get("page_label") or md.get("page")
        path  = md.get("source_path") or md.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        src_lines.append(f"- {title}{page_str} — {path}")
    if src_lines:
        # corta y de-duplica
        src_lines = list(dict.fromkeys([_clip_text(s, 60) for s in src_lines]))[:6]
    src_block = "SOURCES:\n" + ("\n".join(src_lines) if src_lines else "- (no local sources)")

    # 7) Traza y memoria
    _push_turn(state, role="system", name="tactics_system", content=prompt)
    _push_turn(state, role="assistant", name="tactics_advisor", content=md_only)
    _push_turn(state, role="assistant", name="tactics_sources", content=src_block)

    msgs = [
        AIMessage(content=md_only, name="tactics_advisor"),
        AIMessage(content=src_block, name="tactics_sources")
    ]

    # 8) Persistimos en el estado
    state["tactics_md"] = md_only
    state["tactics_struct"] = struct if isinstance(struct, list) else []
    state["tactics_list"] = [ (it.get("name") or "").strip() for it in (struct or []) if isinstance(it, dict) and it.get("name") ]


    #Marca etapa ADD 3.0
    state["arch_stage"] = "TACTICS"        # ahora estamos en la fase de selección de tácticas ADD 3.0
    state["quality_attribute"] = qa        # refuerza cuál atributo estamos atacando
    state["current_asr"] = asr_text        # guarda el ASR que estas tácticas satisfacen

    # Señales para cortar en unifier
    state["endMessage"] = md_only
    state["intent"] = "tactics"
    state["nextNode"] = "unifier"
    prev_msgs = state.get("messages", [])
    return {**state, "messages": prev_msgs + msgs}


def researcher_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    intent = state.get("intent", "general")
    force_rag = bool(state.get("force_rag", False))
    doc_only = bool(state.get("doc_only"))
    ctx_doc = (state.get("doc_context") or "").strip()

    # ⛔ GUARD 1: si estamos en turno ASR y NO se forzó RAG, no investigues
    if intent == "asr" and not force_rag:
        note = "(RAG omitido en turno ASR)" if lang == "es" else "(RAG skipped for ASR turn)"
        _push_turn(state, role="assistant", name="researcher", content=note)
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=note, name="researcher")],
            "hasVisitedInvestigator": True
        }

    # ⛔ GUARD 2: si el intent real es diagrama, este nodo no debe hacer nada
    if intent == "diagram":
        note = "Generando el diagrama con el agente de diagramas…" if lang == "es" else "Diagram will be generated by the diagram agent…"
        _push_turn(state, role="assistant", name="researcher", content=note)
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=note, name="researcher")],
            "hasVisitedInvestigator": True
        }


    # saludo sin RAG
    if intent in ("greeting", "smalltalk"):
        quick = "Hola, ¿en qué tema de arquitectura te gustaría profundizar?" if lang == "es" \
                else "Hi! How can I help you with software architecture today?"
        _push_turn(state, role="assistant", name="researcher", content=quick)
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=quick, name="researcher")],
            "hasVisitedInvestigator": True
        }

    # ---- Agente de investigación (con RAG opcional / DOC-ONLY bloquea RAG) ----
    sys = (
        "You are an expert in software architecture (ADD, quality attributes, tactics, views).\n"
        f"Always reply in {('Spanish' if lang=='es' else 'English')}.\n" +
        ("- If the question is about architecture, you SHOULD call `local_RAG` first to ground your answer, unless force_rag is False."
         if not doc_only else
         "- DOC-ONLY is ON: do NOT call retrieval. Base your answer ONLY on the PROJECT DOCUMENT.\n")
    )

    system_message = SystemMessage(content=sys)
    _push_turn(state, role="system", name="researcher_system", content=sys)

    # Contexto: prioriza doc_context en DOC-ONLY, si no, usa add_context
    ctx_add = (state.get("add_context") or "").strip()
    ctx_for_prompt = ctx_doc if (doc_only and ctx_doc) else ctx_add
    context_message = SystemMessage(
        content=f"PROJECT DOCUMENT (exclusive source):\n{ctx_for_prompt}"
    ) if (doc_only and ctx_for_prompt) else (
        SystemMessage(content=f"PROJECT CONTEXT:\n{ctx_for_prompt}") if ctx_for_prompt else None
    )

    # Herramientas: sin RAG en DOC-ONLY
    tools = ([LLM] + ([LLMWithImages] if _HAS_VERTEX else [])) if doc_only else ([local_RAG, LLM] + ([LLMWithImages] if _HAS_VERTEX else []))
    agent = create_react_agent(llm, tools=tools)

    # HINT corto (solo si no estamos en DOC-ONLY)
    hint_lines = []
    if (force_rag or intent in ("architecture",)) and not doc_only:
        hint_lines.append("Start by calling the tool `local_RAG` with the user's question.")
    if intent == "architecture" and state.get("mermaidCode"):
        hint_lines.append("Also explain the tactics in the provided Mermaid, if any.")
    hint = _clip_text("\n".join(hint_lines).strip(), 100) if hint_lines else ""

    short_history = _last_k_messages(state["messages"], k=6)
    messages_with_system = [system_message] + ([context_message] if context_message else []) + short_history

    payload = {
        "messages": messages_with_system + ([HumanMessage(content=hint)] if hint else []),
        "userQuestion": state["userQuestion"],
        "localQuestion": state["localQuestion"],
        "imagePath1": state["imagePath1"],
        "imagePath2": state["imagePath2"]
    }

    try:
        # Limita la recursión para evitar planeos largos del agente
        result = agent.invoke(payload, config={"recursion_limit": 12})
    except Exception:
        messages_with_system = [system_message] + _last_k_messages(state["messages"], k=3)
        payload["messages"] = messages_with_system
        result = agent.invoke(payload, config={"recursion_limit": 8})

    msgs_out = result.get("messages", [])
    for m in msgs_out:
        _push_turn(state, role="assistant", name="researcher", content=str(getattr(m, "content", m)))
        # NEW: usar la última respuesta del investigador como contexto de negocio/técnico
    if msgs_out:
        last_msg = msgs_out[-1]
        last_text = getattr(last_msg, "content", str(last_msg)) or ""
        # Lo recortamos para no romper el prompt de los siguientes nodos
        state["add_context"] = _clip_text(str(last_text).strip(), 2000)

        return {
        **state,
        "messages": state["messages"] + [
            AIMessage(
                content=str(getattr(m, "content", m)),
                name="researcher"
            ) for m in msgs_out
        ],
        "hasVisitedInvestigator": True
    }



def creator_node(state: GraphState) -> GraphState:
    user_q = state["userQuestion"]
    effective_q = state.get("localQuestion") or user_q

    prompt = f"""{prompt_creator}

User request:
{effective_q}

If an ASR is provided, ensure components and connectors explicitly support the Response and Response Measure.
"""
    _push_turn(state, role="system", name="creator_system", content=prompt)

    response = llm.invoke(prompt)
    content = getattr(response, "content", "")

    match = re.search(r"```mermaid\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    mermaid_code = (match.group(1).strip() if match else "").strip()

    _push_turn(state, role="assistant", name="creator", content=content)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=content, name="creator")],
        "mermaidCode": mermaid_code,
        "hasVisitedCreator": True
    }

def _pick_asr_to_evaluate(state: GraphState) -> str:
    if state.get("last_asr"):
        return state["last_asr"]
    uq = (state.get("userQuestion") or "")
    m = re.search(r"(evaluate|evalúa|evaluar|check|review)\s+(this|este)\s+asr\s*:?\s*(.+)$", uq, re.I | re.S)
    if m:
        return m.group(3).strip()
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and (m.name or "") == "asr_recommender" and m.content:
            return m.content
    return ""

def _book_snippets_for_eval(retriever, concern_hint: str = "") -> str:
    q = "quality attribute scenario parts stimulus source environment artifact response response measure"
    if concern_hint:
        q = concern_hint + " " + q
    try:
        docs = list(retriever.invoke(q))
    except Exception:
        docs = []
    # 4 fragmentos de 300 chars c/u
    seen, out = set(), []
    for d in docs:
        t = (d.page_content or "").strip().replace("\n", " ")
        t = t[:300]
        if t and t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= 4: break
    return "\n\n".join(out)

def getEvaluatorPrompt(image_path1: str, image_path2: str) -> str:
    i1 = f"\nthis is the first image path: {image_path1}" if image_path1 else ""
    i2 = f"\nthis is the second image path: {image_path2}" if image_path2 else ""
    return f"""You are an expert in software-architecture evaluation.
Use:
- Theory Tool (correctness)
- Viability Tool (feasibility)
- Needs Tool (requirements alignment)
- Analyze Tool (compare two diagrams){i1}{i2}
Keep answers short and decisive."""

def evaluator_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    uq = (state.get("userQuestion") or "")
    concern_hint = "latency" if re.search(r"latenc", uq, re.I) else ("scalability" if re.search(r"scalab", uq, re.I) else "")
    doc_only = bool(state.get("doc_only"))
    ctx_doc = (state.get("doc_context") or "").strip()

    # --- MODO 1: evaluación de ASR ---
    if _looks_like_eval(uq):
        asr_text = _pick_asr_to_evaluate(state)
        if not asr_text:
            short = "No encuentro un ASR para evaluar. Pega el texto del ASR o pide que genere uno primero." if lang=="es" \
                    else "I couldn't find an ASR to evaluate. Paste the ASR text or ask me to create one first."
            _push_turn(state, role="assistant", name="evaluator", content=short)
            return {**state, "messages": state["messages"] + [AIMessage(content=short, name="evaluator")], "hasVisitedEvaluator": True}

        if doc_only and ctx_doc:
            book_snips = f"[DOC] {ctx_doc[:1500]}"
        else:
            book_snips = _book_snippets_for_eval(retriever, concern_hint)

        directive = "Responde en español." if lang=="es" else "Answer in English."
        eval_prompt = f"""{directive}
You are evaluating a Quality Attribute Scenario (Architecture Significant Requirement).

BOOK_SNIPPETS (ground your critique in these ideas; keep it short):
{book_snips}

ASR_TO_EVALUATE:
{asr_text}

Write a compact evaluation with EXACTLY these sections (plain text, no Markdown):

Verdict:
  One line: Good / Weak / Invalid, with a short reason.

Gaps:
  3–6 bullets pointing missing or vague parts against the canonical QAS fields (Source, Stimulus, Environment, Artifact, Response, Response Measure).

Quality:
  3–5 bullets about measurability, precision of Response Measure (p95/p99, thresholds), clarity of stimulus, realism.

Risks & Tactics:
  3–5 bullets on plausible risks and which tactics mitigate them (use tactic names verbatim).

Rewrite (improved ASR):
  Provide a tightened ASR using the same QAS structure (Summary, Context, Scenario with the 6 fields, Response Measure). Keep it realistic and measurable.

References:
  List 2–5 short items only if grounded by BOOK_SNIPPETS; otherwise write "None".
"""
        eval_prompt = ("DOC-ONLY mode: ON. Reason exclusively from the PROJECT DOCUMENT.\n\n" + eval_prompt) if doc_only else eval_prompt
        result = llm.invoke(eval_prompt)
        content = getattr(result, "content", str(result)).strip()

        _push_turn(state, role="system", name="evaluator_system", content=eval_prompt)
        _push_turn(state, role="assistant", name="evaluator", content=content)

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=content, name="evaluator")],
            "hasVisitedEvaluator": True
        }

    # --- MODO 2 (fallback): tools variados ---
    tools = [theory_tool, viability_tool, needs_tool]
    if _HAS_VERTEX:
        tools.append(analyze_tool)
    evaluator_agent = create_react_agent(llm, tools=tools)

    eval_prompt = getEvaluatorPrompt(state.get("imagePath1",""), state.get("imagePath2",""))
    ctx_add = (state.get("add_context") or "").strip()[:1500]
    if doc_only and ctx_doc:
        eval_prompt = f"DOC-ONLY: use exclusively this PROJECT DOCUMENT.\n{ctx_doc[:1500]}\n\n" + eval_prompt
    elif ctx_add:
        eval_prompt = f"PROJECT CONTEXT:\n{ctx_add}\n\n" + eval_prompt
    _push_turn(state, role="system", name="evaluator_system", content=eval_prompt)

    messages_with_system = [SystemMessage(content=eval_prompt)] + state["messages"]
    result = evaluator_agent.invoke({
        "messages": messages_with_system,
        "userQuestion": state.get("userQuestion",""),
        "localQuestion": state.get("localQuestion",""),
        "imagePath1": state.get("imagePath1",""),
        "imagePath2": state.get("imagePath2","")
    })

    for msg in result["messages"]:
        _push_turn(state, role="assistant", name="evaluator", content=str(getattr(msg, "content", msg)))

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=msg.content, name="evaluator") for msg in result["messages"]],
        "hasVisitedEvaluator": True
    }

# --- Unifier ---

def _last_ai_by(state: GraphState, name: str) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and getattr(m, "name", None) == name and m.content:
            return m.content
    return ""

def _strip_all_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"^\s*#.*$", "", text, flags=re.M)
    text = text.replace("**", "")
    out = []
    for ln in text.splitlines():
        if re.search(r"^\s*(graph\s+(LR|TB)|flowchart|sequenceDiagram|classDiagram)\b", ln, re.I):
            continue
        if re.match(r"^\s*[A-Za-z0-9_-]+\s*--?[>-]", ln):
            continue
        out.append(ln)
    return "\n".join(out).strip()

def _extract_rag_sources_from(text: str) -> str:
    m = re.search(r"SOURCES:\s*(.+)$", text, flags=re.S | re.I)
    if not m:
        return ""
    raw = m.group(1)
    lines = []
    for ln in raw.splitlines():
        ln = ln.strip(" -\t")
        if ln:
            lines.append(ln)
    return "\n".join(lines[:8])

def _split_sections(text: str) -> dict:
    sections = {"Answer": "", "References": "", "Next": ""}
    current = None
    for ln in text.splitlines():
        if re.match(r"^Answer:", ln, re.I):
            current = "Answer"; sections[current] = ln.split(":", 1)[1].strip(); continue
        if re.match(r"^References:", ln, re.I):
            current = "References"; sections[current] = ln.split(":", 1)[1].strip(); continue
        if re.match(r"^Next:", ln, re.I):
            current = "Next"; sections[current] = ln.split(":", 1)[1].strip(); continue
        if current:
            sections[current] += ("\n" + ln)
    for k in sections:
        sections[k] = sections[k].strip()
    return sections

def unifier_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    intent = state.get("intent", "general")

    # 🟣 NUEVO: caso especial cuando ya tenemos un script Mermaid de diagrama
    mermaid = (state.get("mermaidCode") or "").strip()
    if mermaid:
        if lang == "es":
            head = (
                "Aquí tienes el diagrama solicitado, generado a partir de tu "
                "escenario de calidad (ASR), el estilo arquitectónico seleccionado "
                "y las tácticas priorizadas.\n\n"
                "Puedes copiar y pegar este script Mermaid en tu editor preferido "
                "(por ejemplo, mermaid.live o un plugin de VS Code):\n"
            )
            footer = ""
            suggestions = [
                "Formular un nuevo ASR para otro escenario de calidad.",
                "Generar un diagrama de componentes a partir de este sistema.",
                "Generar un diagrama de despliegue para este mismo sistema.",
            ]
        else:
            head = (
                "Here is the requested diagram, generated from your quality "
                "scenario (ASR), the selected architectural style and the "
                "prioritized tactics.\n\n"
                "You can copy & paste this Mermaid script into your favorite "
                "editor (for example, mermaid.live or a VS Code Mermaid plugin):\n"
            )
            footer = ""
            suggestions = [
                "Generate a new ASR for another quality scenario.",
                "Generate a component diagram from this system.",
                "Generate a deployment diagram for this same system.",
            ]
        
        # Dentro del unifier, rama "if intent == 'diagram' and state.get('mermaidCode')"
        mermaid = state.get("mermaidCode") or ""
        if lang == "es":
            head = "Aquí tienes el diagrama generado a partir del ASR, el estilo y las tácticas seleccionadas."
            footer = ""
            suggestions = [
                "Formular un nuevo ASR para otro escenario de calidad.",
                "Generar un diagrama de componentes a partir de este sistema.",
                "Generar un diagrama de despliegue para este mismo sistema.",
            ]
        else:
            head = (
                "Here is the diagram generated from your quality scenario (ASR), "
                "the selected architecture style and the prioritized tactics."
            )
            footer = ""
            suggestions = [
                "Generate a new ASR for another quality scenario.",
                "Generate a component diagram from this system.",
                "Generate a deployment diagram for this same system.",
            ]

        end_text = head  # 👈 ya NO incluimos el código mermaid en el texto

        state["suggestions"] = suggestions
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        return {**state, "endMessage": end_text, "intent": "diagram"}

    # 0) Mostrar el diagrama si existe (intención "diagram") - LÓGICA ANTIGUA, LA MANTENEMOS
    d = state.get("diagram") or {}
    if d.get("ok") and d.get("svg_b64"):
        data_url = f'data:image/svg+xml;base64,{d["svg_b64"]}'
        if lang == "es":
            head = "Aquí tienes el diagrama solicitado:"
            footer = "¿Qué te gustaría hacer ahora con este diagrama?"
            tips = [
                "Generar un diagrama de componentes a partir de este sistema.",
                "Generar un diagrama de despliegue para este mismo sistema.",
                "Formular un nuevo ASR basado en este sistema.",
            ]
        else:
            head = "Here is your requested diagram:"
            footer = "What would you like to do next with this diagram?"
            tips = [
                "Generate a component diagram from this system.",
                "Generate a deployment diagram for this same system.",
                "Define a new ASR based on this system.",
            ]

        end_text = f"""{head}
![diagram]({data_url})

{footer}
"""
        state["suggestions"] = tips
        return {**state, "endMessage": end_text, "intent": "diagram"}

    # 🔴 Caso especial para ESTILOS
    if intent == "style":
        style_txt = (
            _last_ai_by(state, "style_recommender")
            or state.get("endMessage")
            or "No style content."
        )

        if lang == "es":
            followups = state.get("suggestions") or [
                "Diseña tácticas concretas para este ASR usando el estilo recomendado.",
                "Compárame más a fondo estos dos estilos para este ASR.",
            ]
        else:
            followups = state.get("suggestions") or [
                "Explain concrete tactics for this ASR using the recommended style.",
                "Compare these two styles in more depth for this ASR.",
            ]

        state["suggestions"] = followups
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": style_txt}
        ]
        return {**state, "endMessage": style_txt}

    # 🔴 Caso especial para TÁCTICAS
    if intent == "tactics":
        tactics_md = (
            state.get("tactics_md")
            or _last_ai_by(state, "tactics_advisor")
            or "No tactics content."
        )
        src_txt = _last_ai_by(state, "tactics_sources")
        refs_block = _extract_rag_sources_from(src_txt) if src_txt else "None"

        if lang == "es":
            followups = [
                "Genera un diagrama de componentes aplicando estas tácticas.",
                "Genera un diagrama de despliegue alineado con estas tácticas.",
            ]
            refs_label = "Referencias"
        else:
            followups = [
                "Generate a component diagram applying these tactics.",
                "Generate a deployment diagram aligned with these tactics.",
            ]
            refs_label = "References"

        end_text = f"{tactics_md}\n\n{refs_label}:\n{refs_block}"

        state["suggestions"] = followups
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        return {**state, "endMessage": end_text}

    # 🔴 Caso especial para ASR
    if intent == "asr" or intent == "ASR":
        raw_asr = (
            _last_ai_by(state, "asr_recommender")
            or state.get("endMessage")
            or "No ASR content found for this turn."
        )
        # si el LLM coló tácticas, las quitamos del ASR
        last_asr = _strip_tactics_sections(raw_asr)

        asr_src_txt = _last_ai_by(state, "asr_sources")
        refs_block = _extract_rag_sources_from(asr_src_txt) if asr_src_txt else "None"

        if lang == "es":
            followups = [
                "Propón estilos arquitectónicos para este ASR.",
                "Refina este ASR con métricas y escenarios más específicos.",
            ]
            refs_label = "Referencias"
        else:
            followups = [
                "Propose architecture styles for this ASR.",
                "Refine this ASR with more specific metrics and scenarios.",
            ]
            refs_label = "References"

        end_text = f"{last_asr}\n\n{refs_label}:\n{refs_block}"

        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        state["suggestions"] = followups
        return {**state, "endMessage": end_text}

    # 🔴 Caso especial: saludo / smalltalk
    if intent in ("greeting", "smalltalk"):
        if lang == "es":
            hello = "¡Hola! ¿Sobre qué tema de arquitectura quieres profundizar?"
            nexts = [
                "Formular un ASR (requerimiento de calidad) para mi sistema.",
                "Revisar un ASR que ya tengo.",
            ]
            footer = (
                "Si quieres, podemos empezar el ciclo ADD 3.0 formulando "
                "un ASR (por ejemplo de latencia, disponibilidad o seguridad)."
            )
        else:
            hello = "Hi! What software-architecture topic would you like to explore?"
            nexts = [
                "Define an ASR (quality attribute requirement) for my system.",
                "Review an ASR I already have.",
            ]
            footer = (
                "If you want, we can start the ADD 3.0 cycle by defining "
                "an ASR (for example latency, availability or security)."
            )

        end_text = hello + "\n\n" + footer
        state["suggestions"] = nexts
        return {**state, "endMessage": end_text}

    # 🔵 Caso por defecto: síntesis de investigador / evaluador / etc.
    researcher_txt = _last_ai_by(state, "researcher")
    evaluator_txt = _last_ai_by(state, "evaluator")
    creator_txt = _last_ai_by(state, "creator")
    asr_src_txt = _last_ai_by(state, "asr_sources")

    rag_refs = ""
    if researcher_txt:
        rag_refs = _extract_rag_sources_from(researcher_txt) or ""

    memory_hint = state.get("memory_text", "")

    buckets = []
    if researcher_txt:
        buckets.append(f"researcher:\n{researcher_txt}")
    if evaluator_txt:
        buckets.append(f"evaluator:\n{evaluator_txt}")
    if creator_txt and intent == "diagram":
        buckets.append(f"creator:\n{creator_txt}")
    if asr_src_txt:
        buckets.append(f"asr_sources:\n{asr_src_txt}")

    synthesis_source = (
        "User question:\n"
        + (state.get("userQuestion", ""))
        + "\n\n"
        + "\n\n".join(buckets)
    )

    directive = "Responde en español." if lang == "es" else "Answer in English."
    prompt = f"""{directive}
You are writing the FINAL chat reply.

- Give a complete, direct solution tailored to the question and context.
- Use 6–12 concise lines (bullets or short sentences). No code fences, no mermaid.
- If useful, at the end include a short 'References:' block listing 3–6 items from RAG_SOURCES (one per line). If not useful, you may omit it.

Constraints:
- Use the user's language.
- Do not invent sources outside RAG_SOURCES.
- Keep it clean: no '#', no '**', no code blocks.

Conversation memory (for continuity): {memory_hint}

RAG_SOURCES:
{rag_refs}

SOURCE:
{synthesis_source}
"""

    resp = llm.invoke(prompt)
    final_text = getattr(resp, "content", str(resp))
    final_text = _strip_all_markdown(final_text)

    secs = _split_sections(final_text)
    chips = []
    if secs.get("Next"):
        for ln in secs["Next"].splitlines():
            ln = ln.strip(" -•\t")
            if ln:
                chips.append(ln)
    state["suggestions"] = chips[:6] if chips else []

    _push_turn(state, role="system", name="unifier_system", content=prompt)
    _push_turn(state, role="assistant", name="unifier", content=final_text)

    return {**state, "endMessage": final_text}

# --- Classifier ---

class ClassifyOut(TypedDict):
    language: Literal["en","es"]
    intent: Literal["greeting","smalltalk","architecture","diagram","asr","tactics","style","other"]
    use_rag: bool

def classifier_node(state: GraphState) -> GraphState:
    msg = state.get("userQuestion", "") or ""
    prompt = f"""
Classify the user's last message. Return JSON with:
- language: "en" or "es"
- intent: one of ["greeting","smalltalk","architecture","diagram","asr","tactics","style","other"]
- use_rag: true if this is a software-architecture question (ADD, tactics, latency, scalability,
  quality attributes, views, styles, diagrams, ASR), else false.

User message:
{msg}
"""
    out = llm.with_structured_output(ClassifyOut).invoke(prompt)

    low = msg.lower()
    intent = out["intent"]

    #disparadores de estilo arquitectónico
    style_triggers = [
        "style", "styles",
        "architecture style", "architectural style",
        "estilo", "estilos", "estilo arquitectónico", "estilos arquitectónicos"
    ]
    if any(k in low for k in style_triggers):
        intent = "style"


    tactics_triggers = [
        "tactic", "táctica", "tactica", "tácticas", "tactics", "tactcias",
        "strategy","estrategia",
        "cómo cumplir","como cumplir","how to meet","how to satisfy","how to achieve"
    ]
    if any(k in low for k in tactics_triggers):
        intent = "tactics"
    
    diagram_triggers = [
        "component diagram", "diagram", "diagrama", "diagrama de componentes",
        "uml", "plantuml", "c4", "bpmn",
        "this asr", "este asr", "el asr", "ese asr", "anterior asr"
    ]
    # No pises estilo ni ASR cuando solo dicen "this ASR"
    if any(k in low for k in diagram_triggers) and intent not in ("asr", "style"):
        intent = "diagram"


    return {
        **state,
        "language": out["language"],
        "intent": intent if intent in [
        "greeting",
        "smalltalk",
        "architecture",
        "diagram",
        "asr",
        "tactics",
        "style",
    ] else "general",

        "force_rag": bool(out["use_rag"]),
    }


def boot_node(state: GraphState) -> GraphState:
    """Resetea banderas y buffers al inicio de cada turno (sin borrar last_asr)."""
    return {
        **state,
        "hasVisitedInvestigator": False,
        "hasVisitedCreator": False,
        "hasVisitedEvaluator": False,
        "hasVisitedASR": False,
        "hasVisitedDiagram": False,
        "mermaidCode": "",
        "diagram": {},
        "endMessage": "",
    }

# ========== Wiring

builder.add_node("classifier", classifier_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("investigator", researcher_node)
builder.add_node("creator", creator_node)
builder.add_node("diagram_agent", diagram_orchestrator_node)  # Orquestador
builder.add_node("evaluator", evaluator_node)
builder.add_node("unifier", unifier_node)
builder.add_node("asr", asr_node)
builder.add_node("style", style_node) 
builder.add_node("tactics", tactics_node)


builder.add_node("boot", boot_node)
builder.add_edge(START, "boot")
builder.add_edge("boot", "classifier")
builder.add_edge("classifier", "supervisor")
builder.add_conditional_edges("supervisor", router)
builder.add_edge("investigator", "supervisor")
builder.add_edge("creator", "supervisor")
builder.add_edge("diagram_agent", "supervisor")
builder.add_edge("evaluator", "supervisor")
builder.add_edge("asr", "unifier")
builder.add_edge("style", "unifier")
builder.add_edge("tactics", "unifier")
builder.add_edge("unifier", END)

graph = builder.compile(checkpointer=sqlite_saver)
