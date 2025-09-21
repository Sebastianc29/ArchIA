# ========== Imports 

# Util
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pathlib import Path
import os, re, json, sqlite3
from dotenv import load_dotenv, find_dotenv

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

# GCP (solo si usas vision para comparar diagramas/imagenes)
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import Image

# RAG / citas
from src.rag_agent import get_retriever
from src.quoting import pack_quotes, render_quotes_md

# Kroki
from src.diagram_agent import diagram_node

# --- Token utils (soft) ---
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
    # recorta por caracteres hasta aproximar
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

BASE_DIR = Path(__file__).resolve().parent.parent  # back/
STATE_DIR = BASE_DIR / "state_db"
STATE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = STATE_DIR / "example.db"

conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
sqlite_saver = SqliteSaver(conn)

llm = ChatOpenAI(model="gpt-4o")
retriever = get_retriever()

# ========== Estado

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    userQuestion: str
    localQuestion: str
    hasVisitedInvestigator: bool
    hasVisitedCreator: bool
    hasVisitedEvaluator: bool
    hasVisitedASR: bool
    nextNode: Literal["investigator", "creator", "evaluator", "asr", "unifier"]

    imagePath1: str
    imagePath2: str

    endMessage: str
    mermaidCode: str

    hasVisitedDiagram: bool  # Kroki
    diagram: dict            # output del nodo diagram_agent

    # buffers / RAG / memoria liviana
    turn_messages: list
    retrieved_docs: list
    memory_text: str
    suggestions: list

    # memoria de conversación útil
    last_asr: str  # <— nuevo: último ASR generado en este hilo
    asr_sources_list: list  

    # control de idioma/intención/forcing RAG
    language: Literal["en","es"]
    intent: Literal["general","greeting","smalltalk","architecture","diagram","asr"]
    force_rag: bool


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
    nextNode: Literal["investigator", "creator", "evaluator", "asr", "unifier"]

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
            "enum": ["investigator", "creator", "evaluator", "unifier", "asr"]
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

# ========== Heurísticas de idioma e intenciones de follow-up
# --- ASR evaluation helpers (colócalos cerca de classify_followup) ---
def _looks_like_eval(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in [
        "evaluate this asr", "evalúa este asr", "evaluar este asr",
        "check this asr", "revisa este asr", "review this asr"
    ])

EVAL_TRIGGERS = [
    "evaluate this asr", "check this asr", "review this asr",
    "evalúa este asr", "evalua este asr", "revisa este asr",
    "es bueno este asr", "mejorar este asr", "mejorar asr",
    "critique this asr", "assess this asr"
]

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
# --- ASR evaluate helpers ---
EVAL_TRIGGERS = [
    "evaluate this asr", "check this asr", "review this asr",
    "evalúa este asr", "evalua este asr", "revisa este asr",
    "es bueno este asr", "mejorar este asr", "mejorar asr",
    "critique this asr", "assess this asr"
]

def _wants_asr_evaluation(text: str) -> bool:
    t = (text or "").lower()
    if "asr" in t and any(k in t for k in EVAL_TRIGGERS):
        return True
    # también si el usuario pregunta "can you check/evaluate" y menciona "asr"
    if any(k in t for k in ["evaluate", "check", "review", "evalúa", "evalua", "revisa"]) and "asr" in t:
        return True
    # si pega un “ASR:” explícito
    if "asr:" in t:
        return True
    return False

def _extract_candidate_asr(state: GraphState) -> str:
    """
    1) Si el usuario pegó un ASR explícito en el mensaje, úsalo.
    2) Si no, usa el último ASR generado en este chat (state['last_asr'] o en memory_text).
    3) En última instancia, usa la cadena del usuario (por si escribió una sola frase).
    """
    uq = state.get("userQuestion", "") or ""
    # si viene anotado con ASR:
    m = re.search(r"(?:^|\n)\s*ASR\s*:?(.*)$", uq, flags=re.I | re.S)
    if m:
        cand = m.group(1).strip()
        if cand:
            return cand
    # si hay comillas/líneas cortas, intenta eso
    q_lines = [ln.strip() for ln in uq.splitlines() if ln.strip()]
    if len(q_lines) == 1 and len(q_lines[0]) <= 200:
        return q_lines[0]

    # memoria del turno
    last = state.get("last_asr") or ""
    if last:
        return last

    # memory_text (fallback)
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

    return f"""You are a supervisor orchestrating: investigator, creator (diagrams), evaluator, and ASR advisor.
Choose the next worker and craft a specific sub-question.

Rules:
- If user asks about ADD/architecture, prefer investigator (and it must call local_RAG).
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

# ========== Tools (investigator / RAG / vision)
# ========== Tools (investigator / RAG / vision / evaluator) ==========
from langchain_core.tools import tool

@tool
def LLM(prompt: str) -> dict:
    """Researcher centrado en ADD/ADD 3.0.
    Devuelve un dict con [definition, useCases, examples] según investigatorSchema."""
    return llm.with_structured_output(investigatorSchema).invoke(prompt)

@tool
def LLMWithImages(image_path: str) -> str:
    """Analiza diagramas de arquitectura en imágenes.
    Enfócate en tácticas de performance/disponibilidad y patrones de diseño cuando aplique."""
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
    seen, docs_all = set(), []
    for qq in queries:
        try:
            for d in retriever.invoke(qq):
                # ...
                if len(docs_all) >= 8:  # <-- antes 12
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

# ===== Evaluator tools (con docstrings) =====
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
    """Evalúa corrección teórica vs buenas prácticas (patrones, tácticas, vistas).
    Retorna dict con keys: positiveAspects, negativeAspects, suggestions."""
    return llm.with_structured_output(evaluatorSchema).invoke(
        f"{EVAL_THEORY_PREFIX}\n\nUser input:\n{prompt}"
    )

@tool
def viability_tool(prompt: str) -> dict:
    """Evalúa viabilidad (coste, complejidad, operatividad, riesgos).
    Retorna dict con keys: positiveAspects, negativeAspects, suggestions."""
    return llm.with_structured_output(evaluatorSchema).invoke(
        f"{EVAL_VIABILITY_PREFIX}\n\nUser input:\n{prompt}"
    )

@tool
def needs_tool(prompt: str) -> dict:
    """Valida alineación con necesidades/ASRs y traza decisiones a requerimientos.
    Retorna dict con keys: positiveAspects, negativeAspects, suggestions."""
    return llm.with_structured_output(evaluatorSchema).invoke(
        f"{EVAL_NEEDS_PREFIX}\n\nUser input:\n{prompt}"
    )

@tool
def analyze_tool(image_path: str, image_path2: str) -> str:
    """Compara dos diagramas de arquitectura (p. ej., class vs component) y valora impacto en atributos de calidad."""
    image = Image.load_from_file(image_path)
    image2 = Image.load_from_file(image_path2)
    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    resp = generative_multimodal_model.generate_content([ANALYZE_PREFIX, image, image2])
    return str(resp)


# ========== Router

def router(state: GraphState) -> Literal["investigator", "creator", "evaluator", "diagram_agent", "asr", "unifier"]:
    if state["nextNode"] == "unifier":
        return "unifier"
    elif state["nextNode"] == "asr" and not state.get("hasVisitedASR", False):
        return "asr"
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


# ========== Nodos
def supervisor_node(state: GraphState):
    uq = (state.get("userQuestion") or "")

    # si ya hay un SVG listo en este turno, vamos directo al unifier
    d = state.get("diagram") or {}
    if d.get("ok") and d.get("svg_b64"):
        return {**state, "nextNode": "unifier", "intent": "diagram"}

    # idioma
    lang = detect_lang(uq)
    state_lang = "es" if lang == "es" else "en"

    fu_intent = classify_followup(uq)
    if _looks_like_eval(uq):
        return {**state,
                "localQuestion": uq,      # pasamos el texto tal cual
                "nextNode": "evaluator",
                "intent": "architecture",
                "language": state_lang}
    # keywords para diagramas (ES/EN)
    diagram_terms = [
        "diagrama", "diagrama de componentes", "diagrama de arquitectura",
        "diagram", "component diagram", "architecture diagram",
        "mermaid", "plantuml", "c4", "bpmn", "uml"
    ]
    wants_diagram = any(t in uq.lower() for t in diagram_terms)

    # ✅ renombrado: antes se llamaba `message`
    sys_messages = [SystemMessage(content=makeSupervisorPrompt(state))]

    # baseline LLM (con fallback defensivo)
    try:
        resp = llm.with_structured_output(supervisorSchema).invoke(sys_messages)
        next_node = resp.get("nextNode", "investigator")
        local_q = resp.get("localQuestion", uq)
    except Exception:
        next_node, local_q = "investigator", uq

    intent_val = state.get("intent", "general")
    if any(x in uq.lower() for x in ["asr", "quality attribute scenario", "qas"]) or fu_intent == "make_asr":
        next_node = "asr"; intent_val = "asr"; local_q = f"Create a concrete QAS (ASR) for: {state['userQuestion']}"
    elif wants_diagram or fu_intent in ("component_view","deployment_view","functional_view"):
        next_node = "diagram_agent"; intent_val = "diagram"; local_q = uq
    elif fu_intent in ("explain_tactics","compare","checklist"):
        next_node = "investigator"; intent_val = "architecture"

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

# --- Helpers comunes para ASR ---

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

# --- ASR node ---
def asr_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    uq = state.get("userQuestion", "") or ""

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

    # === Recupera fragmentos de los libros (RAG) ===
    try:
        query = f"{concern} quality attribute scenario latency measure stimulus environment artifact response measure"
        docs_raw = list(retriever.invoke(query))
        docs_list = docs_raw[:6]  # <-- duro tope
    except Exception:
        docs_list = []


    book_snippets = _dedupe_snippets(docs_list, max_items=6, max_chars=800)

    directive = "Answer in English." if lang == "en" else "Responde en español."
    prompt = f"""{directive}
You must write a CONCRETE Quality Attribute Scenario (Architecture Significant Requirement) about **{concern}**.
Plain text only (no Markdown). Use the EXACT section labels below. Be realistic and MEASURABLE.

CRITICAL:
- Ground your choices using the following book snippets. Do NOT invent facts beyond them.
- If a value is not in the snippets, choose reasonable values consistent with the domain, but keep them realistic.
- Output ONLY the ASR sections (no "References", no "Evaluation", no "Next").
- In Response Measure, prefer SLO style (p95/p99 latency, throughput) with concrete numbers.

BOOK_SNIPPETS:
{book_snippets}

Project/domain: {domain}
User input: {uq}

Output with these sections ONLY:

Summary:
  One sentence that captures the {concern} goal for this domain.

Context:
  One short paragraph with business/technical context (users, seasonality, critical paths).

Scenario:
  Source:
  Stimulus:
  Environment:
  Artifact:
  Response:
  Response Measure:

Design tactics to consider:
  - 6–10 tactics explicitly named (e.g., replication, partitioning/sharding, caching, async processing, backpressure, coarse-grained operations, connection pooling, request shedding, load balancing, elastic scaling).

Trade-offs & risks:
  - 3–6 bullets.

Acceptance criteria:
  - 3–5 checks.

Validation plan:
  - 3–5 bullets.
"""
    result = llm.invoke(prompt)
    content = _sanitize_plain_text(getattr(result, "content", str(result)))

    # === Bloque de fuentes (construir SIEMPRE antes de usar) ===
    src_lines = []
    for d in (docs_list or []):
        md = d.metadata or {}
        title = md.get("source_title") or md.get("title") or "doc"
        page  = md.get("page_label") or md.get("page")
        path  = md.get("source_path") or md.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        src_lines.append(f"- {title}{page_str} — {path}")

    # Evita duplicados y limita a 8
        # evita saturar: máximo 4 líneas y cada una capada
    if src_lines:
        src_lines = [ _clip_text(s, 60) for s in src_lines ]  # corta cada línea
        src_lines = list(dict.fromkeys(src_lines))[:4]


    src_block = "SOURCES:\n" + ("\n".join(src_lines) if src_lines else "- (no local sources)")

    # Lista de refs (para memoria/unifier)
    refs_list = []
    for line in src_block.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("sources"):
            continue
        refs_list.append(line.lstrip("- ").strip())

    # Traza para el modal + mensajes
    state["turn_messages"] = state.get("turn_messages", []) + [
        {"role": "system", "name": "asr_system", "content": prompt},
        {"role": "assistant", "name": "asr_recommender", "content": content},
        {"role": "assistant", "name": "asr_sources", "content": src_block},
    ]
    state["messages"] = state["messages"] + [
        AIMessage(content=content, name="asr_recommender"),
        AIMessage(content=src_block, name="asr_sources"),
    ]

    # === MEMORIA DEL CHAT: guarda último ASR + refs ===
    state["last_asr"] = content
    state["asr_sources_list"] = refs_list
    prev_mem = state.get("memory_text", "") or ""
    state["memory_text"] = (prev_mem + f"\n\n[LAST_ASR]\n{content}\n").strip()

    return {**state, "hasVisitedASR": True}

# --- Investigator (researcher) ---

def researcher_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    intent = state.get("intent", "general")
    force_rag = state.get("force_rag", False)

    # saludo sin RAG
    if intent in ("greeting", "smalltalk"):
        quick = "Hola, ¿en qué tema de arquitectura te gustaría profundizar?" if lang == "es" \
                else "Hi! How can I help you with software architecture today?"
        _push_turn(state, role="assistant", name="researcher", content=quick)
        return {**state, "messages": state["messages"] + [AIMessage(content=quick, name="researcher")],
                "hasVisitedInvestigator": True}

    sys = (
        "You are an expert in software architecture (ADD, quality attributes, tactics, views).\n"
        f"Always reply in {('Spanish' if lang=='es' else 'English')}.\n"
        "- If the question is about architecture or ASRs, you MUST call `local_RAG` first "
        "to ground your answer, then optionally complement with LLM or LLMWithImages."
    )
    system_message = SystemMessage(content=sys)
    _push_turn(state, role="system", name="researcher_system", content=sys)

    tools = [local_RAG, LLM, LLMWithImages]
    agent = create_react_agent(llm, tools=tools)

    # HINT corto
    hint_lines = []
    if force_rag or intent in ("architecture", "asr"):
        hint_lines.append("Start by calling the tool `local_RAG` with the user's question.")
    if intent == "architecture" and state.get("mermaidCode"):
        hint_lines.append("Also explain the tactics in the provided Mermaid, if any.")

    hint = _clip_text("\n".join(hint_lines).strip(), 100) if hint_lines else ""

    # <<< CAMBIO CRÍTICO: recorta historial >>>
    short_history = _last_k_messages(state["messages"], k=6)
    messages_with_system = [system_message] + short_history
    payload = {
        "messages": messages_with_system + ([HumanMessage(content=hint)] if hint else []),
        "userQuestion": state["userQuestion"],
        "localQuestion": state["localQuestion"],
        "imagePath1": state["imagePath1"],
        "imagePath2": state["imagePath2"]
    }

    # fallback por rate limit de TPM
    try:
        result = agent.invoke(payload)
    except Exception as e:
        # reintento mínimo: sin hint y con k=3
        messages_with_system = [system_message] + _last_k_messages(state["messages"], k=3)
        payload["messages"] = messages_with_system
        if "messages" in payload and hint:
            payload["messages"] = messages_with_system  # sin hint
        result = agent.invoke(payload)

    for msg in result["messages"]:
        _push_turn(state, role="assistant", name="researcher", content=str(getattr(msg, "content", msg)))

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=msg.content, name="researcher") for msg in result["messages"]],
        "hasVisitedInvestigator": True
    }

# --- Creator ---
def creator_node(state: GraphState) -> GraphState:
    user_q = state["userQuestion"]
    # prefiera localQuestion si trae el ASR embebido por el supervisor
    effective_q = state.get("localQuestion") or user_q

    prompt = f"""{prompt_creator}

User request:
{effective_q}

If an ASR is provided, ensure components and connectors explicitly support the Response and Response Measure.
"""
    _push_turn(state, role="system", name="creator_system", content=prompt)

    response = llm.invoke(prompt)
    content = getattr(response, "content", "")

    # Si sigues con Mermaid:
    match = re.search(r"```mermaid\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    mermaid_code = (match.group(1).strip() if match else "").strip()

    _push_turn(state, role="assistant", name="creator", content=content)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=content, name="creator")],
        "mermaidCode": mermaid_code,
        "hasVisitedCreator": True
    }


# --- Evaluator ---
# --- Helpers para evaluación de ASR (parches mínimos) ---

def _looks_like_eval(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ["evaluate this asr", "evalúa este asr", "evaluar este asr", "check this asr", "revisa este asr", "review this asr"])

def _pick_asr_to_evaluate(state: GraphState) -> str:
    """
    1) Si hay un ASR reciente en memoria del chat, úsalo.
    2) Si el usuario pegó 'Evaluate this ASR: ...', extrae ese bloque.
    3) Fallback: último mensaje del 'asr_recommender' si existe.
    4) Si nada, devuelve cadena vacía.
    """
    # 1) memoria liviana
    if state.get("last_asr"):
        return state["last_asr"]

    # 2) extracción directa desde la pregunta del usuario
    uq = (state.get("userQuestion") or "")
    m = re.search(r"(evaluate|evalúa|evaluar|check|review)\s+(this|este)\s+asr\s*:?\s*(.+)$", uq, re.I | re.S)
    if m:
        return m.group(3).strip()

    # 3) último mensaje del nodo asr_recommender
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage) and (m.name or "") == "asr_recommender" and m.content:
            return m.content

    return ""

def _book_snippets_for_eval(retriever, concern_hint: str = "") -> str:
    """
    Recupera pocas frases cortas para no disparar tokens.
    Limita a 4 fragmentos de 300 chars cada uno.
    """
    q = "quality attribute scenario parts stimulus source environment artifact response response measure"
    if concern_hint:
        q = concern_hint + " " + q
    try:
        docs = list(retriever.invoke(q))
    except Exception:
        docs = []
    return _dedupe_snippets(docs, max_items=4, max_chars=300)  # ya tienes _dedupe_snippets arriba

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
    """
    Modo 1 (nuevo): si el usuario pide evaluar un ASR, hacemos evaluación directa, usando RAG,
    y devolvemos una respuesta compacta con veredicto, gaps y una versión reescrita del ASR.

    Modo 2 (fallback): si NO es evaluación de ASR, usamos tu agente anterior (theory/viability/needs/analyze)
    tal cual para no romper nada.
    """
    lang = state.get("language", "es")
    uq = (state.get("userQuestion") or "")
    concern_hint = "latency" if re.search(r"latenc", uq, re.I) else ("scalability" if re.search(r"scalab", uq, re.I) else "")

    # --- MODO 1: evaluación de ASR ---
    if _looks_like_eval(uq):
        asr_text = _pick_asr_to_evaluate(state)
        if not asr_text:
            short = "No encuentro un ASR para evaluar. Pega el texto del ASR o pide que genere uno primero." if lang=="es" \
                    else "I couldn't find an ASR to evaluate. Paste the ASR text or ask me to create one first."
            _push_turn(state, role="assistant", name="evaluator", content=short)
            return {**state, "messages": state["messages"] + [AIMessage(content=short, name="evaluator")], "hasVisitedEvaluator": True}

        # RAG: trocitos cortos de los libros para fundamentar
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
        result = llm.invoke(eval_prompt)
        content = getattr(result, "content", str(result)).strip()

        # traza para el modal + mensajes
        _push_turn(state, role="system", name="evaluator_system", content=eval_prompt)
        _push_turn(state, role="assistant", name="evaluator", content=content)

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=content, name="evaluator")],
            "hasVisitedEvaluator": True
        }

    # --- MODO 2 (fallback): comportamiento original con tools ---
    evaluator_agent = create_react_agent(llm, tools=[theory_tool, viability_tool, needs_tool, analyze_tool])

    eval_prompt = getEvaluatorPrompt(state.get("imagePath1",""), state.get("imagePath2",""))
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

def _last_ai_by(state, name: str) -> str:
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

    # 1) Caso especial: ASR -> entrega tal cual + referencias + next
    if intent == "asr":
        last_asr = _last_ai_by(state, "asr_recommender") or "No ASR content found for this turn."
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
    

    # luego construyes synthesis_source como ya lo tienes

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
    intent: Literal["greeting","smalltalk","architecture","diagram","asr","other"]
    use_rag: bool

def classifier_node(state: GraphState) -> GraphState:
    msg = state.get("userQuestion", "") or ""
    prompt = f"""
Classify the user's last message. Return JSON with:
- language: "en" or "es"
- intent: one of ["greeting","smalltalk","architecture","diagram","asr","other"]
- use_rag: true if this is a software-architecture question (ADD, tactics, latency, scalability,
  quality attributes, views, styles, diagrams, ASR), else false.

User message:
{msg}
"""
    out = llm.with_structured_output(ClassifyOut).invoke(prompt)

    low = msg.lower()
    intent = out["intent"]
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
        "intent": intent if intent in ["greeting","smalltalk","architecture","diagram","asr"] else "general",
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
        # last_asr se conserva para que el siguiente turno pueda usarlo
    }


# ========== Wiring

builder.add_node("classifier", classifier_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("investigator", researcher_node)
builder.add_node("creator", creator_node)
builder.add_node("diagram_agent", diagram_node)  # Kroki
builder.add_node("evaluator", evaluator_node)
builder.add_node("unifier", unifier_node)
builder.add_node("asr", asr_node)

builder.add_node("boot", boot_node)
builder.add_edge(START, "boot")
builder.add_edge("boot", "classifier")
builder.add_edge("classifier", "supervisor")
builder.add_conditional_edges("supervisor", router)
builder.add_edge("investigator", "supervisor")
builder.add_edge("creator", "supervisor")
builder.add_edge("diagram_agent", "supervisor")  # Kroki
builder.add_edge("evaluator", "supervisor")
builder.add_edge("asr", "supervisor")
builder.add_edge("unifier", END)

graph = builder.compile(checkpointer=sqlite_saver)
