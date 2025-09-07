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

def router(state: GraphState) -> Literal["investigator", "creator", "evaluator", "asr", "unifier"]:
    if state["nextNode"] == "unifier":
        return "unifier"
    elif state["nextNode"] == "asr" and not state.get("hasVisitedASR", False):
        return "asr"
    elif state["nextNode"] == "investigator" and not state["hasVisitedInvestigator"]:
        return "investigator"
    elif state["nextNode"] == "creator" and not state["hasVisitedCreator"]:
        return "creator"
    elif state["nextNode"] == "evaluator" and not state["hasVisitedEvaluator"]:
        return "evaluator"
    else:
        return "unifier"

# ========== Nodos

def supervisor_node(state: GraphState):
    uq = (state.get("userQuestion") or "")
    # idioma preferido
    lang = detect_lang(uq)
    state_lang = "es" if lang == "es" else "en"

    # follow-ups explícitos (no creamos un nodo aparte para evitar errores)
    fu_intent = classify_followup(uq)

    # Detecciones duras
    asr_terms = ["asr", "quality attribute scenario", "qas"]
    wants_asr = any(t in uq.lower() for t in asr_terms) or ("asr example" in uq.lower())
    diagram_terms = ["diagram", "mermaid", "draw", "component diagram", "architecture diagram"]
    wants_diagram = any(t in uq.lower() for t in diagram_terms) or bool(state.get("imagePath1") or state.get("imagePath2"))

    # Baseline por LLM
    message = [{"role": "system", "content": makeSupervisorPrompt(state)}]
    response = llm.with_structured_output(supervisorSchema).invoke(message)
    next_node = response["nextNode"]
    local_q = response["localQuestion"]

    # Reglas prioritarias
    intent_val = state.get("intent", "general")
    if wants_asr:
        next_node = "asr"
        intent_val = "asr"
        local_q = f"Create a concrete QAS (ASR) for: {state['userQuestion']}"
    elif fu_intent == "make_asr":
        next_node = "asr"
        intent_val = "asr"
        local_q = f"Create a concrete QAS (ASR) for: {state['userQuestion']}"
    elif wants_diagram or fu_intent in ("component_view","deployment_view","functional_view"):
        next_node = "creator"
        intent_val = "diagram"
        local_q = uq
    elif fu_intent in ("explain_tactics","compare","checklist"):
        next_node = "investigator"
        intent_val = "architecture"

    # Evitar ir a unifier sin nada
    if next_node == "unifier" and not (
        state["hasVisitedInvestigator"] or state["hasVisitedCreator"] or
        state["hasVisitedEvaluator"] or state.get("hasVisitedASR", False)
    ):
        next_node = "investigator"
        intent_val = "architecture"

    return {
        **state,
        "localQuestion": local_q,
        "nextNode": next_node,
        "intent": intent_val,
        "language": state_lang,
    }

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

def _last_ai_by(state, name: str) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and getattr(m, "name", None) == name and m.content:
            return m.content
    return ""

def _strip_markdown_and_diagrams(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)   # cualquier fence
    text = re.sub(r"^\s*#.*$", "", text, flags=re.M)    # titulos
    text = text.replace("**", "")
    # quitar líneas típicas de mermaid
    out = []
    for ln in text.splitlines():
        if re.search(r"^\s*(graph\s+(LR|TB)|flowchart|sequenceDiagram|classDiagram)\b", ln, re.I):
            continue
        if re.match(r"^\s*[A-Za-z0-9_-]+\s*--?[>-]", ln):  # edges
            continue
        out.append(ln)
    return "\n".join(out).strip()

def unifier_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    intent = state.get("intent", "general")

    # Si viene de ASR, solo adjuntamos y proponemos follow-ups
    if intent == "asr":
        last_asr = _last_ai_by(state, "asr_recommender") or "No ASR content found for this turn."
        followups = [
            "Want a component diagram for this?",
            "See a Functional view next?",
            "Check a Deployment view and hosting options?",
            "Compare scalability vs. latency tactics?",
            "Would you like an example ASR for this case?",
            "Map quality drivers → tactics for your context?",
            "Do you want architectural styles to consider?",
        ]
        end_text = last_asr + "\n\nSuggested follow-ups (English):\n- " + "\n- ".join(followups)
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        return {**state, "endMessage": end_text, "suggestions": followups}

    # Mezcla de fuentes
    buckets = []
    for agent_name in ["researcher", "evaluator", "asr_evaluator", "asr_recommender", "creator"]:
        if agent_name == "creator" and intent != "diagram":
            continue
        c = _last_ai_by(state, agent_name)
        if c:
            buckets.append(f"{agent_name} says:\n{c}")

    synthesis_source = "User question:\n" + (state.get("userQuestion","")) + "\n\n" + "\n\n".join(buckets)

    seed = [
        "Want a component diagram for this?",
        "See a Functional view next?",
        "Check a Deployment view and hosting options?",
        "Compare scalability vs. latency tactics?",
        "Would you like an example ASR for this case?",
        "Map quality drivers → tactics for your context?",
        "Do you want architectural styles to consider?",
    ]

    directive = "Answer in English." if lang == "en" else "Responde en español."
    prompt = f"""{directive}
Write a compact but complete synthesis using ONLY the SOURCE. Plain text only (no Markdown, no code fences, no mermaid).

Format and labels EXACTLY:

Summary:
  2–4 concise lines tailored to the question.

Development:
  - 5–10 bullets or short paragraphs with concrete, actionable details from SOURCE.

Decisions & trade-offs:
  - 3–6 bullets with explicit trade-offs and when/when-not.

Checklist:
  - 4–8 short, testable items.

Citations:
  - If SOURCE referenced identifiable documents, list them as: <title> (p.X) — <short path or id>.

Suggested follow-ups (English):
  - 5–8 one-line questions practitioners often ask next, choose from or adapt:
    {", ".join(seed)}

Rules:
- Ignore any diagram-like snippets unless intent is 'diagram'.
- Do not repeat earlier turns verbatim.
- Write in the user's language for all sections except the follow-ups, which stay in English.

=== SOURCE BEGIN ===
{synthesis_source}
=== SOURCE END ===
"""
    _push_turn(state, role="system", name="unifier_system", content=prompt)
    resp = llm.invoke(prompt)
    final_text = getattr(resp, "content", str(resp))
    final_text = _strip_markdown_and_diagrams(final_text) if intent != "diagram" else final_text
    _push_turn(state, role="assistant", name="unifier", content=final_text)

    # extrae follow-ups (si vienen formateados) para el front; si no, usa seed
    suggs = []
    for m in re.findall(r"Suggested follow-ups.*?:\s*(?:- .+?)(?:\n\n|\Z)", final_text, re.I | re.S):
        suggs += [s.strip("- ").strip() for s in m.splitlines() if s.strip().startswith("- ")]
    if not suggs:
        suggs = seed

    return {**state, "endMessage": final_text, "suggestions": suggs}

# --- Classifier ---

class ClassifyOut(TypedDict):
    language: Literal["en","es"]
    intent: Literal["greeting","smalltalk","architecture","diagram","asr","other"]
    use_rag: bool

def classifier_node(state: GraphState) -> GraphState:
    msg = state.get("userQuestion", "")
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

    low = (msg or "").lower()
    intent = out["intent"]
    if ("diagram" in low or "mermaid" in low) and intent != "asr":
        intent = "diagram"

    return {
        **state,
        "language": out["language"],
        "intent": intent if intent in ["greeting","smalltalk","architecture","diagram","asr"] else "general",
        "force_rag": bool(out["use_rag"]),
    }

# ========== Wiring

builder.add_node("classifier", classifier_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("investigator", researcher_node)
builder.add_node("creator", creator_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("unifier", unifier_node)
builder.add_node("asr", asr_node)

builder.add_edge(START, "classifier")
builder.add_edge("classifier", "supervisor")
builder.add_conditional_edges("supervisor", router)
builder.add_edge("investigator", "supervisor")
builder.add_edge("creator", "supervisor")
builder.add_edge("evaluator", "supervisor")
builder.add_edge("asr", "supervisor")
builder.add_edge("unifier", END)

graph = builder.compile(checkpointer=sqlite_saver)
