from __future__ import annotations
# ========== Imports

# Util
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pathlib import Path
import os, re, json, sqlite3, base64, logging
from dotenv import load_dotenv, find_dotenv

# HTTP
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

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

BASE_DIR = Path(__file__).resolve().parent.parent  # back/
STATE_DIR = BASE_DIR / "state_db"
STATE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = STATE_DIR / "example.db"

conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
sqlite_saver = SqliteSaver(conn)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
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

def _extract_first_json_block(md: str):
    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", md, re.S|re.I)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

# ========== PlantUML helper (LLM)

PLANTUML_SYSTEM = """You are an expert software architect and PlantUML author.
Convert the user's natural-language request into a PlantUML DEPLOYMENT diagram.

HARD RULES:
- Output ONLY PlantUML between @startuml and @enduml (no prose, no code fences).
- ASCII only (no « », →, ↔, …). Use <<stereotypes>> and -> arrows.
- Prefer compact structure: cloud "Internet", node "k8s cluster" as cluster, package "namespace" { ... }.
- Use 'database' for DBs, 'queue' for brokers if needed, 'folder' for volumes.
- Prefer relationships even if the user didn't specify them explicitly (infer reasonable ones):
  Ingress -> Services; Services -> Deployments/Pods; Deployments -> DB/Cache; Workers -> DB/Cache/Queues.
- Annotate ports and replicas in labels when provided (e.g., "api (8000) replicas=3").
- Keep output readable.
"""

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
    url = f"{DIAGRAM_BASE}/diagram/nl"
    try:
        r = _HTTP.post(url, json={"prompt": uq, "output_format": fmt}, timeout=(5, 180))
        if r.status_code == 200:
            b64 = base64.b64encode(r.content).decode("ascii")
            ctype = r.headers.get("content-type", "image/svg+xml")
            return {"ok": True, "svg_b64": b64, "content_type": ctype}
        return {"ok": False, "error": f"{r.status_code}: {r.text[:200]}"}
    except requests.Timeout as e:
        return {"ok": False, "error": f"NL backend timeout: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"NL backend error: {e}"}

def _render_puml_with_backend(puml: str, fmt: str = DIAGRAM_FORMAT) -> dict:
    url = f"{DIAGRAM_BASE}/diagram/render"
    payload = {"diagram_type": "plantuml", "source": puml, "output_format": fmt}
    try:
        r = _HTTP.post(url, json=payload, timeout=(5, 180))
        if r.status_code == 200:
            b64 = base64.b64encode(r.content).decode("ascii")
            ctype = r.headers.get("content-type", "image/svg+xml")
            return {"ok": True, "svg_b64": b64, "content_type": ctype}
        return {"ok": False, "error": f"{r.status_code}: {r.text[:200]}"}
    except requests.Timeout as e:
        return {"ok": False, "error": f"Render backend timeout: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"Render backend error: {e}"}

# In-proc (opcional)
def _render_inproc_puml(puml: str, fmt: str = DIAGRAM_FORMAT) -> dict:
    """Render en-proceso si tienes un cliente local disponible (evita HTTP loopback)."""
    if not DIAGRAM_INPROC:
        return {"ok": False, "error": "in-proc disabled"}
    try:
        # Si tienes un cliente local para Kroki/PlantUML, con esta interfaz:
        # from src.clients.kroki_client import render_plantuml
        from src.clients.kroki_client import render_plantuml  # type: ignore
        svg_bytes = render_plantuml(puml, fmt=fmt)  # debe devolver bytes
        b64 = base64.b64encode(svg_bytes).decode("ascii")
        return {"ok": True, "svg_b64": b64, "content_type": "image/svg+xml"}
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
    nextNode: Literal["investigator", "creator", "evaluator", "diagram_agent", "tactics", "asr", "unifier"]
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
    intent: Literal["general","greeting","smalltalk","architecture","diagram","asr","tactics"]
    force_rag: bool

    # etapa actual del pipeline ASR -> tacticas -> despliegue
    arch_stage:str
    quality_attribute:str
    add_context:str
    tactics_list:list
    current_asr: str #ASR vigente
    tactics_md: str #salida markdonw del tactics_node
    tactics_struct: list #salida JSON parseada del tactics_node

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
    nextNode: Literal["investigator", "creator", "evaluator", "diagram_agent", "tactics", "asr", "unifier"]

supervisorSchema = {
    "title": "SupervisorResponse",
    "description": "Response from the supervisor indicating the next node and the setup question.",
    "type": "object",
    "properties": {
        "localQuestion": {"type": "string", "description": "What is the question for the worker node?"},
        "nextNode": {
            "type": "string",
            "description": "The next node to act.",
            "enum": ["investigator", "creator", "evaluator", "unifier", "asr", "diagram_agent", "tactics"]
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

def router(state: GraphState) -> Literal["investigator","creator","evaluator","diagram_agent","tactics","asr","unifier"]:
    if state["nextNode"] == "unifier":
        return "unifier"
    elif state["nextNode"] == "asr" and not state.get("hasVisitedASR", False):
        return "asr"
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

def diagram_orchestrator_node(state: "GraphState") -> "GraphState":
    """
    Orquesta: NL parser del backend  y/o LLM->PlantUML->render.
    - DIAGRAM_NL_MODE=off:       solo backend /diagram/nl
    - DIAGRAM_NL_MODE=always:    siempre LLM -> /diagram/render (fallback a /nl si falta flechas)
    - DIAGRAM_NL_MODE=fallback:  intenta /nl; si falla, usa LLM -> /render (y como 3er fallback, in-proc si existe)
    """
    uq = state.get("localQuestion") or state.get("userQuestion") or ""
    mode = DIAGRAM_NL_MODE

    def _ok(res: dict) -> bool:
        return bool(res and res.get("ok") and res.get("svg_b64"))

    # 1) Modo off → solo NL backend
    if mode == "off":
        res = _render_nl_with_backend(uq)
        state["diagram"] = res
        state["hasVisitedDiagram"] = True
        return state

    # 2) Modo always → LLM→PUML→render (fallback a NL y/o in-proc)
    if mode == "always":
        try:
            puml = _llm_nl_to_puml(uq)
            if ("->" not in puml) and ("-->" not in puml):
                res = _render_nl_with_backend(uq)
                if not _ok(res) and DIAGRAM_INPROC:
                    res = _render_inproc_puml(puml)
            else:
                res = _render_puml_with_backend(puml)
                if not _ok(res) and DIAGRAM_INPROC:
                    res = _render_inproc_puml(puml)
                if not _ok(res):
                    res2 = _render_nl_with_backend(uq)
                    if _ok(res2): res = res2
        except Exception as e:
            log.warning("LLM->PUML failed: %s", e)
            res = _render_nl_with_backend(uq)
        state["diagram"] = res
        state["hasVisitedDiagram"] = True
        return state

    # 3) Modo fallback (recomendado)
    nl = _render_nl_with_backend(uq)
    if _ok(nl):
        state["diagram"] = nl
        state["hasVisitedDiagram"] = True
        return state

    # NL falló → LLM→PUML
    try:
        puml = _llm_nl_to_puml(uq)
        res = _render_puml_with_backend(puml)
        if not _ok(res) and DIAGRAM_INPROC:
            res = _render_inproc_puml(puml)
    except Exception as e:
        res = {"ok": False, "error": f"LLM/Render error: {e}"}

    state["diagram"] = res
    state["hasVisitedDiagram"] = True
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

    # 1) ASR primero (no incluir tácticas aquí)
    if any(x in uq.lower() for x in ["asr", "quality attribute scenario", "qas"]) or fu_intent == "make_asr":
        next_node = "asr"
        intent_val = "asr"
        local_q = f"Create a concrete QAS (ASR) for: {state['userQuestion']}"

    # 2) DIAGRAMA cuando lo piden
    elif wants_diagram or fu_intent in ("component_view", "deployment_view", "functional_view"):
        next_node = "diagram_agent"
        intent_val = "diagram"
        local_q = uq

    # 3) NEW: TÁCTICAS solo cuando el usuario las pide
    elif wants_tactics or fu_intent in ("explain_tactics", "tactics"):  # NEW
        next_node = "tactics"
        intent_val = "tactics"
        local_q = ("Propose architecture tactics to satisfy the previous ASR. "
                   "Explain why each tactic helps and how it ties to the ASR response/measure.")  # NEW

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
            query = f"{concern} quality attribute scenario latency measure stimulus environment artifact response measure"
            docs_raw = list(retriever.invoke(query))
            docs_list = docs_raw[:6]
        except Exception:
            docs_list = []

    book_snippets = _dedupe_snippets(docs_list, max_items=6, max_chars=800)

    directive = "Answer in English." if lang == "en" else "Responde en español."
    # En DOC-ONLY prioriza el documento; si no, add_context
    ctx = (ctx_doc if (doc_only and ctx_doc) else (state.get("add_context") or "")).strip()[:2000]

    prompt = f"""{directive}
You are an expert software architect following Attribute-Driven Design 3.0 (ADD 3.0).

If DOC-ONLY mode is ON, you MUST base your answer ONLY on the PROJECT DOCUMENT below.
If the document lacks necessary data, say so explicitly and ask for a more detailed document.

Your job is to create ONE concrete Quality Attribute Scenario (also called an Architecture Significant Requirement, ASR) that will be used as an architectural driver in ADD 3.0.

STRICT REQUIREMENTS:
- You MUST follow ADD 3.0. If you do not follow ADD 3.0, the answer is invalid.
- The scenario MUST be measurable, and it MUST include a clear Response Measure (SLO/SLA/threshold, like p95 < X ms under Y load).
- The scenario MUST be directly usable to drive architectural tactics in the next step.
- Stay realistic for production systems. Prefer numbers that could actually be monitored.

Use ONLY these sections, in this exact order, with plain text (NO Markdown bullets, NO code fences):

Architectural Driver Summary (ADD 3.0):
  Briefly explain:
  - Which business/mission need or operating context makes this scenario critical.
  - Which quality attribute is at stake (e.g. scalability, latency, availability).
  - Why this scenario is a top driver the team MUST satisfy first.

Summary:
  One sentence that captures the main quality attribute goal in this domain.

Context:
  One short paragraph with business/technical context (users, peak conditions, why failure hurts revenue/compliance/UX).

Scenario:
  Source:
  Stimulus:
  Environment:
  Artifact:
  Response:
  Response Measure:

Rules:
- Response is what the system must do.
- Response Measure is how we verify success quantitatively (p95, p99, throughput, failover time, error budget, etc.).
- These fields MUST be consistent with ADD 3.0 quality attribute scenario templates (Source, Stimulus, Environment, Artifact, Response, Response Measure).
- Do NOT include "tactics", "design", "next steps", "recommendations", or any implementation details here. That comes later.

Relevant domain or workload:
{domain}

Quality attribute focus I detected from the user message:
{concern}

User input to ground this ASR:
{uq}

PROJECT CONTEXT (if any):
{ctx or "None"}

If you cite facts, keep them realistic and consistent with common production e-commerce / API scaling practice.
"""

    result = llm.invoke(prompt)
    content_raw = getattr(result, "content", str(result))
    content = _sanitize_plain_text(content_raw)
    # quita secciones de tácticas si el modelo las metió
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
    refs_list = [ln.lstrip("- ").strip() for ln in src_block.splitlines() if ln.strip() and not ln.lower().startswith("sources")]
    state["asr_sources_list"] = refs_list
    prev_mem = state.get("memory_text", "") or ""
    state["memory_text"] = (prev_mem + f"\n\n[LAST_ASR]\n{content}\n").strip()

    # Exponer metadata clave del ASR para que main.py la persista (ojo al typo)
    state["quality_attribute"] = concern           # corregido
    state["arch_stage"] = "ASR"                   # estamos en la etapa de definir driver ADD 3.0
    state["current_asr"] = content                # copia directa del ASR final

    # Señales para cortar el turno y NO volver a investigar
    state["endMessage"] = content
    state["hasVisitedASR"] = True
    state["force_rag"] = False
    state["nextNode"] = "unifier"                      # cierra en unifier

    return state

def _strip_code_block_json(md: str) -> str:
    # Remueve el primer bloque ```json ... ``` si existe (dejamos solo el Markdown legible)
    return re.sub(r"```json\s.*?```", "", md, flags=re.S | re.I).strip()

def _guess_quality_attribute(text: str) -> str:
    low = (text or "").lower()
    if "latenc" in low or "response time" in low: return "latency"
    if "scalab" in low or "throughput" in low:    return "scalability"
    if "availab" in low or "uptime" in low:       return "availability"
    if "secur" in low:                             return "security"
    if "modifiab" in low or "change" in low:       return "modifiability"
    if "reliab" in low or "fault" in low:          return "reliability"
    return "performance"

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


GROUNDING (use ONLY this context; if DOC-ONLY, this is the exclusive source):
{book_snippets or "(none)"}

If DOC-ONLY is ON, do not rely on knowledge beyond the PROJECT DOCUMENT even if you “know” typical tactics. If the document does not support a tactic, state “not supported by the document”.

You MUST output THREE sections, in EXACT order:

(0) Why this ASR matters (ADD 3.0 driver):
- 3–5 concise lines.
- Explain why this ASR is critical for system success, including user experience, revenue, compliance, or SLO/SLA obligations.
- Explicitly link back to the ASR's Stimulus, Response, and Response Measure. Example: "If we get a 10x traffic spike (Stimulus) we must keep checkout under 200ms p95 (Response Measure), or we lose conversion."

(1) TACTICS:
List 6–10 concrete architectural tactics we should consider FIRST in ADD 3.0 to satisfy this ASR.
For EACH tactic include:
- Name — canonical architecture tactic name (e.g. "Elastic Horizontal Scaling", "Bulkhead Isolation", "Edge Caching", "Circuit Breaker", "Request Throttling / Rate Limiting", "Async Queue + Worker").
- Rationale — why THIS tactic directly helps satisfy THIS ASR's Response and Response Measure.
- Consequences / Trade-offs — cost, complexity, operational risk, coupling, vendor lock-in, debugging difficulty, blast radius, etc.
- When to use — trigger condition in runtime terms (for example: "Use this if traffic spikes 10x in <5 min and you MUST keep p95 checkout under 200ms and avoid cascading failures").

(2) JSON:
Finally output ONE ```json code fence with an array of objects like:
[
  {{
    "name": "Elastic Horizontal Scaling",
    "purpose": "Keep p95 checkout latency under 200ms during 10x traffic bursts",
    "rationale": "We spin up N replicas automatically so incoming requests never queue long enough to violate the Response Measure",
    "risks": ["Higher infra spend at peak", "Needs autoscaling rules tuned", "Can expose noisy-neighbor issues if isolation is weak"],
    "tradeoffs": ["More cost vs better resilience at peak"],
    "categories": ["scalability","latency","availability"],
    "traces_to_asr": "Stimulus=10x burst, Response=scale out, Response Measure=p95 < 200ms under burst",
    "expected_effect": "Checkout stays responsive and revenue is not lost during peak"
  }}
]

STRICT RULES:
- You MUST behave like ADD 3.0: tactics are chosen BECAUSE OF the ASR's Response and Response Measure, not randomly.
- Every tactic MUST explicitly tie back to the ASR driver.
- DO NOT invent product names or vendor SKUs. Stay pattern-level.
- Keep output concise, production-realistic, and auditable.
"""

    resp = llm.invoke(prompt)
    raw = getattr(resp, "content", str(resp)).strip()

    # 5) Parseamos JSON y preparamos salidas
    struct = _extract_first_json_block(raw) or []
    md_only = _strip_code_block_json(raw)

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

    return {**state, "messages": state["messages"] + msgs}


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

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=getattr(m, "content", str(m)), name="researcher") for m in msgs_out],
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

    # Mostrar el diagrama si existe
    d = state.get("diagram") or {}
    if d.get("ok") and d.get("svg_b64"):
        data_url = f'data:image/svg+xml;base64,{d["svg_b64"]}'
        head = "Aquí tienes el diagrama solicitado:" if lang=="es" else "Here is your requested diagram:"
        tips = [
            "¿Quieres el mismo diagrama en PNG?" if lang=="es" else "Do you want this diagram as PNG?",
            "¿Genero también una vista de despliegue?" if lang=="es" else "Generate a Deployment view too?",
            "¿Deseas ver/editar el código fuente?" if lang=="es" else "Want to see/edit the diagram source?"
        ]
        end_text = f"""{head}
![diagram]({data_url})

Next:
- """ + "\n- ".join(tips)
        state["suggestions"] = tips
        return {**state, "endMessage": end_text, "intent": "diagram"}
    
    # Caso especial: TÁCTICAS
    if intent == "tactics":
        tactics_md = state.get("tactics_md") or _last_ai_by(state, "tactics_advisor") or "No tactics content."
        src_txt = _last_ai_by(state, "tactics_sources")
        refs_block = _extract_rag_sources_from(src_txt) if src_txt else "None"

        followups = [
            "¿Genero un diagrama de componentes aplicando estas tácticas?" if lang=="es" else "Generate a component diagram applying these tactics?",
            "¿Convertimos estas tácticas en criterios de pruebas no funcionales?" if lang=="es" else "Turn these tactics into non-functional test criteria?",
            "¿Quieres una estimación de riesgos/costos por táctica?" if lang=="es" else "Estimate risks/costs per tactic?",
            "¿Mapeamos táctica → componente/servicio concreto?" if lang=="es" else "Map tactic → concrete component/service?"
        ]
        end_text = f"{tactics_md}\n\nReferences:\n{refs_block}\n\nNext:\n- " + "\n- ".join(followups)

        state["suggestions"] = followups
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        return {**state, "endMessage": end_text}


    # 1) Caso especial: ASR
    if intent == "asr":
        raw_asr = _last_ai_by(state, "asr_recommender") or "No ASR content found for this turn."
        #si el LLM coló tácticas, las quitamos del ASR
        last_asr = _strip_tactics_sections(raw_asr)

        asr_src_txt = _last_ai_by(state, "asr_sources")
        refs_block = _extract_rag_sources_from(asr_src_txt) if asr_src_txt else "None"

        followups = [
            "¿Quieres un diagrama de componentes específico para este ASR?" if lang=="es" else "Want a component diagram for THIS ASR?",
            "¿Validamos despliegue/hosting para cumplir el ASR?" if lang=="es" else "Check a Deployment/hosting plan to meet the ASR?",
            "¿Comparamos latencia vs. escalabilidad para este dominio?" if lang=="es" else "Compare latency vs scalability for this domain?",
            "¿Convertimos este ASR en criterios de pruebas?" if lang=="es" else "Turn this ASR into test criteria?",
        ]

        end_text = (
            f"{last_asr}\n\n"
            f"References:\n{refs_block}\n\n"
            "Next:\n- " + "\n- ".join(followups)
        )

        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        state["suggestions"] = followups
        return {**state, "endMessage": end_text}


    # 2) Caso especial: saludo / smalltalk
    if intent in ("greeting", "smalltalk"):
        hello = "¡Hola! ¿Sobre qué tema de arquitectura quieres profundizar?" if lang=="es" \
                else "Hi! What software-architecture topic would you like to explore?"
        nexts = [
            "Latency tactics vs scalability?",
            "Create a deployment view for my web app?",
            "Give me a scalability ASR example?",
            "Compare event-driven vs request/response?",
        ]
        end_text = hello + "\n\nNext:\n- " + "\n- ".join(nexts)
        state["suggestions"] = nexts
        return {**state, "endMessage": end_text}

    # 3) Compacto por defecto
    researcher_txt = _last_ai_by(state, "researcher")
    evaluator_txt  = _last_ai_by(state, "evaluator")
    creator_txt    = _last_ai_by(state, "creator")
    asr_src_txt    = _last_ai_by(state, "asr_sources")

    rag_refs = ""
    if researcher_txt:
        rag_refs = _extract_rag_sources_from(researcher_txt) or ""

    memory_hint = state.get("memory_text", "")

    buckets = []
    if researcher_txt: buckets.append(f"researcher:\n{researcher_txt}")
    if evaluator_txt:  buckets.append(f"evaluator:\n{evaluator_txt}")
    if creator_txt and intent == "diagram": buckets.append(f"creator:\n{creator_txt}")
    if asr_src_txt: buckets.append(f"asr_sources:\n{asr_src_txt}")

    synthesis_source = "User question:\n" + (state.get("userQuestion","")) + "\n\n" + "\n\n".join(buckets)

    directive = "Responde en español." if lang=="es" else "Answer in English."
    prompt = f"""{directive}
You are writing the FINAL chat reply. Produce **compact output** with three sections ONLY and no Markdown headers:

Answer:
- Give a complete, direct solution tailored to the question and context.
- 6–12 concise lines (bullets or short sentences). No code fences, no mermaid.

References:
- If RAG_SOURCES has entries, list 3–6 relevant items (short, one per line). If empty, write "None".

Next:
- 3–5 short, context-aware follow-up questions to continue THIS conversation.

Constraints:
- Use the user's language for all sections.
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
            if ln: chips.append(ln)
    state["suggestions"] = chips[:6] if chips else state.get("suggestions", [])

    _push_turn(state, role="system", name="unifier_system", content=prompt)
    _push_turn(state, role="assistant", name="unifier", content=final_text)

    return {**state, "endMessage": final_text}

# --- Classifier ---

class ClassifyOut(TypedDict):
    language: Literal["en","es"]
    intent: Literal["greeting","smalltalk","architecture","diagram","asr","tactics","other"]
    use_rag: bool

def classifier_node(state: GraphState) -> GraphState:
    msg = state.get("userQuestion", "") or ""
    prompt = f"""
Classify the user's last message. Return JSON with:
- language: "en" or "es"
- intent: one of ["greeting","smalltalk","architecture","diagram","asr","tactics","other"]
- use_rag: true if this is a software-architecture question (ADD, tactics, latency, scalability,
  quality attributes, views, styles, diagrams, ASR), else false.

User message:
{msg}
"""
    out = llm.with_structured_output(ClassifyOut).invoke(prompt)

    low = msg.lower()
    intent = out["intent"]
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
    if any(k in low for k in diagram_triggers) and intent != "asr":
        intent = "diagram"

    return {
        **state,
        "language": out["language"],
        "intent": intent if intent in ["greeting","smalltalk","architecture","diagram","asr","tactics"] else "general",
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
builder.add_edge("tactics", "unifier")
builder.add_edge("unifier", END)

graph = builder.compile(checkpointer=sqlite_saver)
