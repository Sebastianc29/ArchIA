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

from langchain_core.tools import tool

@tool
def LLM(prompt: str) -> str:
    """Researcher restricted to Attribute-Driven Design (ADD/ADD 3.0).
    Returns a structured answer with fields [definition, useCases, examples] using investigatorSchema."""
    response = llm.with_structured_output(investigatorSchema).invoke(prompt)
    return response


@tool
def LLMWithImages(image_path: str) -> str:
    """Researcher that analyzes software architecture diagrams in images.
    Focus on performance/availability tactics and OOD patterns when applicable."""
    image = Image.load_from_file(image_path)
    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    response = generative_multimodal_model.generate_content([
        "What software architecture tactics can you see in this diagram? "
        "If it is a class diagram, analyze and evaluate it by identifying classes, attributes, methods, relationships, "
        "Object-Oriented Design principles, and design patterns.",
        image
    ])
    return response


@tool
def local_RAG(prompt: str) -> str:
    """Answer about performance & scalability using local documents (RAG).
    Returns a short synthesis followed by a SOURCES block for UI consumption."""
    q = (prompt or "").strip()

    synonyms = []
    if re.search(r"\badd\b", q, re.I):
        synonyms += [
            "Attribute-Driven Design",
            "ADD 3.0",
            "architecture design method ADD",
            "Bass Clements Kazman ADD",
            "quality attribute scenarios ADD",
        ]
    if re.search(r"scalab|latenc|throughput|performance|tactic", q, re.I):
        synonyms += [
            "performance and scalability tactics",
            "latency tactics",
            "scalability tactics",
            "architectural tactics performance",
        ]

    queries = [q] + [f"{q} — {s}" for s in synonyms]
    seen, docs_all = set(), []

    for qq in queries:
        try:
            for d in retriever.invoke(qq):
                key = (d.metadata.get("source_path") or d.metadata.get("source"), d.metadata.get("page"))
                if key in seen:
                    continue
                seen.add(key)
                docs_all.append(d)
                if len(docs_all) >= 12:
                    break
        except Exception:
            pass
        if len(docs_all) >= 12:
            break

    if not docs_all:
        return "No local documents were retrieved for this query."

    preview = []
    for i, d in enumerate(docs_all[:4], 1):
        snip = (d.page_content or "").replace("\n", " ").strip()
        snip = (snip[:700] + "…") if len(snip) > 700 else snip
        preview.append(f"[{i}] {snip}")

    src_lines = []
    for d in docs_all:
        title = d.metadata.get("title") or Path(d.metadata.get("source_path", "")).stem or "doc"
        page = d.metadata.get("page_label") or d.metadata.get("page")
        src = d.metadata.get("source_path") or d.metadata.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        src_lines.append(f"- {title}{page_str} — {src}")

    preview_text = "\n\n".join(preview)
    sources_text = "\n".join(src_lines)

    return f"""{preview_text}

SOURCES:
{sources_text}
"""


# ===== Evaluator

@tool
def theory_tool(prompt: str) -> str:
    """Evaluator that checks the theoretical correctness of an architecture against best practices.
    Returns structured fields [positiveAspects, negativeAspects, suggestions]."""
    response = llm.with_structured_output(evaluatorSchema).invoke(theory_prompt + prompt)
    return response


@tool
def viability_tool(prompt: str) -> str:
    """Evaluator that analyzes the feasibility/viability of the proposed strategy.
    Returns structured fields [positiveAspects, negativeAspects, suggestions]."""
    response = llm.with_structured_output(evaluatorSchema).invoke(viability_prompt + prompt)
    return response


@tool
def needs_tool(prompt: str) -> str:
    """Evaluator that checks alignment with user needs/requirements.
    Returns structured fields [positiveAspects, negativeAspects, suggestions]."""
    response = llm.with_structured_output(evaluatorSchema).invoke(needs_prompt + prompt)
    return response


@tool
def analyze_tool(image_path: str, image_path2: str) -> str:
    """Evaluator that compares a class diagram vs a component diagram for a component,
    assessing support for quality attributes (e.g., scalability/performance)."""
    image = Image.load_from_file(image_path)
    image2 = Image.load_from_file(image_path2)
    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    response = generative_multimodal_model.generate_content([
        analyze_prompt,
        image,
        image2
    ])
    return response

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

    # keywords para diagramas (ES/EN)
    diagram_terms = [
        "diagrama", "diagrama de componentes", "diagrama de arquitectura",
        "diagram", "component diagram", "architecture diagram",
        "mermaid", "plantuml", "c4", "bpmn", "uml"
    ]
    wants_diagram = any(t in uq.lower() for t in diagram_terms)

    # baseline LLM (no incluye diagram_agent en el schema, está ok)
    message = [{"role": "system", "content": makeSupervisorPrompt(state)}]
    response = llm.with_structured_output(supervisorSchema).invoke(message)
    next_node = response["nextNode"]
    local_q = response["localQuestion"]

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

    return {**state, "localQuestion": local_q, "nextNode": next_node, "intent": intent_val, "language": state_lang}


# --- ASR node ---

def _sanitize_plain_text(txt: str) -> str:
    txt = re.sub(r"```.*?```", "", txt, flags=re.S)
    txt = txt.replace("**", "")
    txt = re.sub(r"^\s*#\s*", "", txt, flags=re.M)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def asr_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    uq = state.get("userQuestion", "")

    concern = "scalability" if re.search(r"scalab", uq, re.I) else \
              "latency" if re.search(r"latenc", uq, re.I) else "performance"

    low = uq.lower()
    if "e-comm" in low or "commerce" in low or "shop" in low:
        domain = "e-commerce flash sale"
    elif "api" in low:
        domain = "public REST API with burst traffic"
    elif "stream" in low or "kafka" in low:
        domain = "event streaming pipeline"
    else:
        domain = "e-commerce flash sale"

    tactics_snippets = ""
    try:
        docs = retriever.invoke(f"{concern} tactics site:local")
        parts = []
        for d in docs[:4]:
            parts.append((d.page_content or "")[:600])
        tactics_snippets = "\n\n".join(parts)
    except Exception:
        pass

    directive = "Answer in English." if lang == "en" else "Responde en español."
    prompt = f"""{directive}
Write a concrete Architecture Significant Requirement as a Quality Attribute Scenario for {concern}.
Plain text only (no Markdown).

Project/domain: {domain}
User input: {uq}

Use EXACTLY these labeled sections and fill with realistic, measurable values:

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
  - 6–10 tactics explicitly named (replication, partitioning/sharding, caching, async processing, backpressure, coarse-grained operations, connection pooling, request shedding, load balancing, elastic scaling).

Trade-offs & risks:
  - 3–6 bullets.

Acceptance criteria:
  - 3–5 checks.

Validation plan:
  - 3–5 bullets.

Optional context (quote tactic names verbatim):
{tactics_snippets}
"""
    result = llm.invoke(prompt)
    content = _sanitize_plain_text(getattr(result, "content", str(result)))

    message = AIMessage(content=content, name="asr_recommender")
    state["turn_messages"] = state.get("turn_messages", []) + [
        {"role": "system", "name": "asr_system", "content": prompt},
        {"role": "assistant", "name": "asr_recommender", "content": content},
    ]

    return {**state, "messages": state["messages"] + [message], "hasVisitedASR": True}

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

    # Hint para orientar al agente
    hint_lines = []
    if force_rag or intent in ("architecture", "asr"):
        hint_lines.append("Start by calling the tool `local_RAG` with the user's question.")
    # Si el usuario dijo "explain tactics" y tenemos un mermaid previo, pásalo
    if intent in ("architecture",) and re.search(r"tactic|táctica|explain|explica", state.get("userQuestion",""), re.I):
        if state.get("mermaidCode"):
            hint_lines.append("Explain the tactics used in this Mermaid diagram:\n```mermaid\n" + state["mermaidCode"] + "\n```")

    hint = "\n".join(hint_lines).strip()
    messages_with_system = [system_message] + state["messages"]
    payload = {"messages": messages_with_system + ([HumanMessage(content=hint)] if hint else []),
               "userQuestion": state["userQuestion"],
               "localQuestion": state["localQuestion"],
               "imagePath1": state["imagePath1"],
               "imagePath2": state["imagePath2"]}

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
    prompt = f"{prompt_creator}\n\nUser request:\n{user_q}"
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

# --- Evaluator ---

def getEvaluatorPrompt(image_path1: str, image_path2: str) -> str:
    i1 = f"\nthis is the first image path: {image_path1}" if image_path1 else ""
    i2 = f"\nthis is the second image path: {image_path2}" if image_path2 else ""
    return f"""You are an expert in software-architecture evaluation.
Use:
- Theory Tool (correctness)
- Viability Tool (feasibility)
- Needs Tool (requirements alignment)
- Analyze Tool (compare two diagrams){i1}{i2}
"""

def evaluator_node(state: GraphState) -> GraphState:
    evaluator_agent = create_react_agent(llm, tools=[theory_tool, viability_tool, needs_tool, analyze_tool])

    eval_prompt = getEvaluatorPrompt(state["imagePath1"], state["imagePath2"])
    _push_turn(state, role="system", name="evaluator_system", content=eval_prompt)

    messages_with_system = [SystemMessage(content=eval_prompt)] + state["messages"]
    result = evaluator_agent.invoke({
        "messages": messages_with_system,
        "userQuestion": state["userQuestion"],
        "localQuestion": state["localQuestion"],
        "imagePath1": state["imagePath1"],
        "imagePath2": state["imagePath2"]
    })

    for msg in result["messages"]:
        _push_turn(state, role="assistant", name="evaluator", content=str(getattr(msg, "content", msg)))

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=msg.content, name="evaluator") for msg in result["messages"]],
        "hasVisitedEvaluator": True
    }

# --- Unifier ---
# ===== Unifier (compact answer) =====
import re
from langchain_core.messages import AIMessage

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

    # 🖼️ Mostrar el diagrama si existe
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


    # --- (resto de tu unifier tal cual) ---


    # 1) Caso especial: ASR
    if intent == "asr":
        last_asr = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, AIMessage) and (m.name or "") == "asr_recommender" and m.content:
                last_asr = m.content
                break
        if not last_asr:
            last_asr = "No ASR content found for this turn."
        followups = [
            "¿Quieres un diagrama de componentes para este ASR?" if lang=="es" else "Want a component diagram for this?",
            "¿Validamos el despliegue y hosting?" if lang=="es" else "Check a Deployment view and hosting options?",
            "¿Comparamos latencia vs. escalabilidad?" if lang=="es" else "Compare latency vs. scalability?",
        ]
        end_text = f"{last_asr}\n\nNext:\n- " + "\n- ".join(followups)
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

    # 3) Compacto por defecto (respuesta + referencias + siguientes pasos)
    researcher_txt = _last_ai_by(state, "researcher")
    evaluator_txt  = _last_ai_by(state, "evaluator")
    creator_txt    = _last_ai_by(state, "creator")

    rag_refs = _extract_rag_sources_from(researcher_txt) if researcher_txt else ""
    memory_hint = state.get("memory_text", "")

    buckets = []
    if researcher_txt: buckets.append(f"researcher:\n{researcher_txt}")
    if evaluator_txt:  buckets.append(f"evaluator:\n{evaluator_txt}")
    if creator_txt and intent == "diagram": buckets.append(f"creator:\n{creator_txt}")

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

    # ← clave: disparar 'diagram' ante “diagrama/diagram/mermaid/uml…”
    if any(k in low for k in ["diagrama", "diagram", "mermaid", "plantuml", "c4", "bpmn", "uml"]):
        if intent != "asr":
            intent = "diagram"

    return {
        **state,
        "language": out["language"],
        "intent": intent if intent in ["greeting","smalltalk","architecture","diagram","asr"] else "general",
        "force_rag": bool(out["use_rag"]),
    }


def boot_node(state: GraphState) -> GraphState:
    """Resetea banderas y buffers al inicio de cada turno."""
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