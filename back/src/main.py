# src/main.py

from typing import Optional
from pathlib import Path
import os, re, sqlite3, base64

from fastapi import UploadFile, File, Form, HTTPException, Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager


from langchain_core.messages import HumanMessage
from src.graph import graph
from src.rag_agent import create_or_load_vectorstore
from src.memory import (
    init as memory_init,
    get as memory_get,
    set_kv as memory_set,
    load_arch_flow,
    save_arch_flow,
)
from src.services.doc_ingest import extract_pdf_text
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
memory_init()

# ===================== Detección simple de idioma (ES/EN) ==========================
def detect_lang(q: str) -> str:
    ql = (q or "").lower()
    if re.search(r"[áéíóúñ¿¡]", ql): return "es"
    if re.search(r"\b(what|how|why|when|which|where|who|the|and|or|if|is|are|can|do|does|should|would)\b", ql): return "en"
    ascii_ratio = sum(1 for c in q if ord(c) < 128) / max(1, len(q))
    return "en" if ascii_ratio > 0.97 else "es"

# ===================== Lifespan ==========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        create_or_load_vectorstore()
        print("[startup] RAG listo")
    except Exception as e:
        print(f"[startup] RAG init omitido: {e}")
    yield
    print("[shutdown] Cerrando app...")

# Una sola instancia de FastAPI
app = FastAPI(title="ArquIA API", lifespan=lifespan)

# ===================== Paths ==========================
BACK_DIR = Path(__file__).resolve().parent.parent  # .../back/
IMAGES_DIR = BACK_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = BACK_DIR / "docs_uploads"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_DIR = BACK_DIR / "feedback_db"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DB_PATH = FEEDBACK_DIR / "feedback.db"

# ===================== DB Feedback ======================
def init_feedback_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(FEEDBACK_DB_PATH), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS message_feedback (
            session_id   TEXT NOT NULL,
            message_id   INTEGER NOT NULL,
            thumbs_up    INTEGER DEFAULT 0,
            thumbs_down  INTEGER DEFAULT 0,
            PRIMARY KEY (session_id, message_id)
        )
        """
    )
    conn.commit()
    return conn

feedback_conn = init_feedback_db()

def get_next_message_id(session_id: str) -> int:
    cur = feedback_conn.cursor()
    cur.execute("SELECT MAX(message_id) FROM message_feedback WHERE session_id = ?", (session_id,))
    row = cur.fetchone()
    return (row[0] or 0) + 1

def upsert_feedback(session_id: str, message_id: int, up: int = 0, down: int = 0):
    feedback_conn.execute(
        "INSERT OR REPLACE INTO message_feedback (session_id, message_id, thumbs_up, thumbs_down) VALUES (?, ?, ?, ?)",
        (session_id, message_id, up, down),
    )
    feedback_conn.commit()

def update_feedback(session_id: str, message_id: int, up: int, down: int):
    feedback_conn.execute(
        "UPDATE message_feedback SET thumbs_up = ?, thumbs_down = ? WHERE session_id = ? AND message_id = ?",
        (up, down, session_id, message_id),
    )
    feedback_conn.commit()

# ===================== CORS ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== Helpers de tema ===================
def _normalize_topic(xx: str) -> str:
    x = (xx or "").lower()
    if "latenc" in x or "latencia" in x: return "latency"
    if "scalab" in x or "escalabilidad" in x: return "scalability"
    if "availab" in x or "disponibilidad" in x: return "availability"
    if "perform" in x or "rendim" in x: return "performance"
    return ""

def _extract_topic_from_text(q: str) -> str:
    return _normalize_topic(q)

def _needs_topic_hint(q: str) -> bool:
    low = (q or "").lower()
    mentions_tactics = bool(re.search(r"\btactic|\btáctic|\btactica|\btáctica", low))
    has_topic = bool(_extract_topic_from_text(low))
    return mentions_tactics and not has_topic

# ===================== ASR helpers =======================
ASR_HEAD_RE = re.compile(
    r"\b(ASR|Architecture[-\s]?Significant[-\s]?Requirement|Requisit[oa]\s+Significativ[oa]\s+de\s+Arquitectura)\b[:：]?",
    re.I,
)

def _looks_like_make_asr(msg: str) -> bool:
    if not msg: return False
    low = msg.lower()
    return bool(re.search(r"\b(create|make|draft|write|generate|produce|compose)\b.*\b(asr)\b", low)) \
        or bool(re.search(r"\b(crea|haz|redacta|genera|produce)\b.*\b(asr)\b", low))

def _extract_asr_from_message(msg: str) -> str:
    if not msg: return ""
    m = re.search(r"(?:review|evaluate|revisar|evalua\w*)\s+(?:this|este|esta)?\s*asr\s*[:：]\s*(.+)$", msg, re.I | re.S)
    if m: return m.group(1).strip()
    m = re.search(r"(?:review|evaluate|revisar|evalua\w*)\s+(?:this|este|esta)?\s*asr[\s,)\-:]*\s*(.+)$", msg, re.I | re.S)
    if m: return m.group(1).strip()
    m = re.search(r"\basr\s*[:：]\s*(.+)$", msg, re.I | re.S)
    if m: return m.group(1).strip()
    m = re.search(r"(?:review|evaluate|revisar|evalua\w*)\s+(?:this|este|esta)?\s*asr\s*\((.+)\)", msg, re.I | re.S)
    if m: return m.group(1).strip()
    return ""

def _extract_asr_from_result_text(text: str) -> str:
    if not text: return ""
    m = re.search(r"```asr\s*([\s\S]*?)```", text, re.I)
    if m: return m.group(1).strip()
    if re.search(r"\b(Summary|Context|Design\s+tactics|Trade[-\s]?offs|Acceptance\s+criteria|Validation\s+plan)\b", text, re.I):
        return text.strip()
    m = ASR_HEAD_RE.search(text)
    if m:
        start = m.start()
        asr = text[start:]
        asr = re.split(r"\n\s*#{1,6}\s|\n\s*(?:Rationale|Razonamiento|Conclusiones)\s*[:：]", asr, maxsplit=1)[0]
        return asr.strip()
    m = re.search(r"(?:^|\n)\s*[-*]\s*ASR\s*[:：]\s*(.+)", text, re.I)
    if m: return m.group(1).strip()
    return ""

def _wants_diagram_of_that_asr(msg: str) -> bool:
    if not msg: return False
    low = msg.lower()
    wants_diagram = any(k in low for k in ["diagram", "diagrama"])
    mentions_component = any(k in low for k in ["component diagram", "diagram component", "componentes", "de componentes"])
    mentions_that_asr = any(k in low for k in ["that asr", "ese asr", "esa asr", "dicho asr", "ese requisito"])
    return (wants_diagram and mentions_that_asr) or (mentions_component and mentions_that_asr)

def _wants_style(txt: str) -> bool:
    low = (txt or "").lower()
    keys = [
        # EN
        "architecture style",
        "architectural style",
        "style for this asr",
        "styles for this asr",
        "style for the asr",
        "what style", "which style",
        # ES
        "estilo de arquitectura",
        "estilo arquitectónico",
        "estilos para este asr",
        "qué estilo", "que estilo",
    ]
    # también capturamos frases donde simplemente se combinan "style" y "asr"
    return any(k in low for k in keys) or ("style" in low and "asr" in low)


def _wants_tactics(txt: str) -> bool:
    low = (txt or "").lower()
    keys = [
        "táctica", "tactica", "tácticas", "tacticas",
        "tactic", "tactics",
        "estrategia", "estrategias", "strategy", "strategies",
        "cómo cumplir", "como cumplir",
        "how to satisfy", "how to meet", "how to achieve"
    ]
    return any(k in low for k in keys)

def _wants_deployment(txt: str) -> bool:
    low = (txt or "").lower()
    keys = [
        "despliegue", "deployment", "deployment diagram",
        "diagrama de despliegue",
        "plantuml", "mermaid"
    ]
    return any(k in low for k in keys)


# ===================== Health ===========================
@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ===================== /message =========================
@app.post("/message")
async def message(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(...),
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
):
    if not message:
        raise HTTPException(status_code=400, detail="No message provided")
    if not session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")

    # Identidad simple por sesión
    user_id = request.headers.get("X-User-Id") or session_id
    arch_flow = load_arch_flow(user_id)

    # ID incremental para feedback por mensaje
    message_id = get_next_message_id(session_id)

    # Usar un thread_id POR SESION, no por mensaje
    thread_id = session_id

    # --- Adjuntos (imagen o PDF) ---
    def _is_pdf(up):
        return bool(up and up.filename and (
            (up.content_type or "").lower().startswith("application/pdf")
            or up.filename.lower().endswith(".pdf")
        ))

    async def _save(up, dst_dir):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", up.filename or "file")
        p = dst_dir / f"{session_id}__{safe}"
        with open(p, "wb") as f:
            f.write(await up.read())
        return p

    image_path1, image_path2 = "", ""
    doc_context, doc_only = "", False

    # image1
    if image1 and image1.filename:
        if _is_pdf(image1):
            p = await _save(image1, DOCS_DIR)
            doc_context = extract_pdf_text(str(p), max_chars=8000) or ""
            doc_only = bool(doc_context.strip())
        else:
            p = await _save(image1, IMAGES_DIR)
            image_path1 = str(p)

    # image2
    if image2 and image2.filename:
        if _is_pdf(image2):
            p = await _save(image2, DOCS_DIR)
            extra = extract_pdf_text(str(p), max_chars=8000) or ""
            doc_context = (doc_context + "\n\n" + extra).strip() if extra else doc_context
            doc_only = bool(doc_context.strip())
        else:
            p = await _save(image2, IMAGES_DIR)
            image_path2 = str(p)

    # --- Turno actual como HumanMessage(s) ---
    turn_messages = [HumanMessage(content=message)]
    if image_path1:
        turn_messages.append(HumanMessage(content=f"[image_path_1] {image_path1}"))
    if image_path2:
        turn_messages.append(HumanMessage(content=f"[image_path_2] {image_path2}"))
    if doc_only and doc_context:
        # visible en el turno para trazabilidad
        turn_messages.append(HumanMessage(content=f"[DOCUMENT_EXCERPT]\n{doc_context[:4000]}"))

    # --- Memoria previa (MEJORADA) ---
    last_topic = memory_get(user_id, "topic", "")

    # ➜ FIX: antes se usaba uploaded_pdf_snippets (no existe). Usamos doc_context.
    pdf_context_turn = doc_context  # FIX

    if pdf_context_turn:
        # Persistimos en arch_flow.add_context (append no destructivo)
        af = dict(arch_flow)
        prev_ctx = (af.get("add_context") or "").strip()
        af["add_context"] = (prev_ctx + "\n\n" + pdf_context_turn).strip() if prev_ctx else pdf_context_turn
        save_arch_flow(user_id, af)
        arch_flow = af  # usarlo ya mismo

    memory_text = (
        f"Stage: {arch_flow.get('stage','')}\n"
        f"Quality Attribute: {arch_flow.get('quality_attribute','')}\n"
        f"Business / Context: {arch_flow.get('add_context','')}\n"
        f"Current ASR:\n{arch_flow.get('current_asr','')}\n\n"
        f"Architecture style: {arch_flow.get('style','')}\n"
        f"Tactics so far: {arch_flow.get('tactics', [])}\n"
        f"User last topic: {last_topic}"
    ).strip() or "N/A"

    # --- ASR pegado por el usuario (si lo hay) ---
    asr_in_msg = _extract_asr_from_message(message)
    if asr_in_msg:
        memory_set(user_id, "current_asr", asr_in_msg)
    made_asr = _looks_like_make_asr(message)

    # --- Config del grafo ---
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
    user_lang = detect_lang(message)

    # --- Heurísticas locales ---
    topic_hint = _extract_topic_from_text(message) or _extract_topic_from_text(last_topic)
    msg_low = message.lower()
    force_rag = (
        _needs_topic_hint(message) or
        bool(re.search(
            r"\b(add|qas|asr|tactic|táctica|latenc|scalab|throughput|rendim|availability|disponib|diagrama|diagram)\b",
            msg_low
        ))
    )

    if doc_only:
        force_rag = False  # DOC-ONLY desactiva RAG

    user_intent = "general"
    if not arch_flow.get("current_asr"):
        # Si aún no hay ASR, cualquier cosa va a ASR primero
        user_intent = "asr"
    elif _wants_style(message):
        # Ya hay ASR y el usuario está pidiendo estilos
        user_intent = "style"
    elif _wants_tactics(message):
        user_intent = "tactics"
    elif _wants_deployment(message):
        user_intent = "diagram"


    # --- Limpieza parcial del estado (sin borrar historial persistente del grafo) ---
    try:
        graph.update_state(config, {"values": {
            "endMessage": "",
            "mermaidCode": "",
            "diagram": {},  # FIX: dict vacío, no None
            "hasVisitedDiagram": False,
            "turn_messages": [],
            "current_asr": memory_get(user_id, "current_asr", ""),
        }})
    except Exception:
        pass

    # --- Invocación del grafo ---
    try:
        result = graph.invoke(
            {
                "messages": turn_messages,
                "userQuestion": message,
                "localQuestion": "",
                "hasVisitedInvestigator": False,
                "hasVisitedCreator": False,
                "hasVisitedEvaluator": False,
                "hasVisitedASR": False,
                "nextNode": "supervisor",
                "imagePath1": image_path1,
                "imagePath2": image_path2,
                "doc_only": doc_only,
                "doc_context": doc_context,
                "endMessage": "",
                "mermaidCode": "",
                "turn_messages": [],
                "retrieved_docs": [],
                "memory_text": memory_text,  # memoria rica
                "suggestions": [],
                "language": user_lang,
                "intent": user_intent,
                "force_rag": force_rag,
                "topic_hint": topic_hint,  # opcional; el grafo puede ignorarlo
                "current_asr": memory_get(user_id, "current_asr", ""),
                "style": arch_flow.get("style", ""),
                "selected_style": arch_flow.get("style", ""),
                "last_style": arch_flow.get("style", ""),
                "arch_stage": arch_flow.get("stage", ""),
                "quality_attribute": arch_flow.get("quality_attribute", ""),
                "add_context": arch_flow.get("add_context", ""),
                "tactics_list": arch_flow.get("tactics", []),
            },
            config,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph error: {e}")

    # --- Feedback inicial ---
    upsert_feedback(session_id=session_id, message_id=message_id, up=0, down=0)

    # --- Actualiza memoria simple ---
    low = message.lower()
    if "latencia" in low:
        memory_set(user_id, "topic", "latencia")
    elif "escalabilidad" in low:
        memory_set(user_id, "topic", "escalabilidad")
    if "asr" in low:
        memory_set(user_id, "asr_notes", message)

    # --- Captura ASR desde la respuesta del grafo (si redactó uno) ---
    end_msg = result.get("endMessage", "") or ""
    asr_from_result = _extract_asr_from_result_text(end_msg)
    if asr_from_result:
        memory_set(user_id, "current_asr", asr_from_result)
    elif made_asr and len(end_msg) > 80:
        memory_set(user_id, "current_asr", end_msg.strip())

    # Actualizar arch_flow con el ASR generado/refinado
    if result.get("hasVisitedASR"):
        arch_flow["current_asr"] = memory_get(user_id, "current_asr", "")
        arch_flow["quality_attribute"] = result.get(
            "asr_quality_attribute",
            arch_flow.get("quality_attribute", "")
        )
        arch_flow["add_context"] = result.get(
            "asr_context",
            arch_flow.get("add_context", "")
        )
        arch_flow["stage"] = "ASR"

    style_text = (
    result.get("style")
    or result.get("selected_style")
    or result.get("last_style")
    )

    if style_text and result.get("arch_stage") == "STYLE":
        arch_flow["style"] = style_text
        arch_flow["stage"] = "STYLE"


    # --- Persistir tácticas si este turno fue de tácticas ---
    tactics_json = result.get("tactics_struct") or None
    tactics_md   = result.get("tactics_md") or ""
    if user_intent == "tactics" and (tactics_json or tactics_md):
        arch_flow["tactics"] = tactics_json or []
        arch_flow["stage"] = "TACTICS"

        # ===================== DIAGRAMA =====================
    # Ya no generamos SVG/PNG ni usamos Kroki.
    # Solo devolvemos el script de Mermaid que construye el grafo con ASR + estilo + tácticas.
    diagram_obj = {}

    # Si el usuario pidió explícitamente un diagrama de despliegue, marcamos el stage
    if _wants_deployment(message):
        arch_flow["stage"] = "DEPLOYMENT"

    # Persistimos el flujo ADD 3.0 actualizado (ASR, estilo, tácticas, stage, etc.)
    save_arch_flow(user_id, arch_flow)

    # Mermaid generado por el grafo (diagram_orchestrator_node)
        # Mermaid generado por el grafo (diagram_orchestrator_node)
    mermaid_code = (result.get("mermaidCode") or "").strip()

    # Siempre que tengamos Mermaid, pegamos el bloque ```mermaid``` al final.
    # (Da igual si el intent que vimos era "diagram" o no.)
    if mermaid_code:
        mermaid_help = (
            "\n\n---\n"
            "Here is the **Mermaid script** for this diagram.\n"
            "You can copy & paste it into the Mermaid live editor (https://mermaid.live), "
            "a VS Code Mermaid plugin, or any compatible renderer:\n\n"
            "```mermaid\n"
            f"{mermaid_code}\n"
            "```"
        )
        end_msg = (end_msg + mermaid_help).strip()

    else:
        # Aseguramos que end_msg esté definido
        end_msg = end_msg.strip()

    # --- Payload al front (no pisamos suggestions si las necesitas) ---
    clean_payload = {
        "endMessage": end_msg,
        "mermaidCode": mermaid_code,
        "diagram": diagram_obj,  # ahora siempre vacío; ya no mandamos SVG
        "messages": result.get("turn_messages", []),
        "session_id": session_id,
        "message_id": message_id,
        "thread_id": thread_id,
        "suggestions": result.get("suggestions", []),
    }

    return clean_payload

    # --- Payload al front (no pisamos suggestions si las necesitas) ---
    clean_payload = {
        "endMessage": end_msg,
        "mermaidCode": mermaid_code,
        "diagram": diagram_obj,                # ahora puede venir vacío si user_intent == "diagram"
        "messages": result.get("turn_messages", []),
        "session_id": session_id,
        "message_id": message_id,
        "thread_id": thread_id,
        "suggestions": result.get("suggestions", []),
    }

    return clean_payload


# ===================== /feedback ========================
@app.post("/feedback")
async def feedback(
    session_id: str = Form(...),
    message_id: int = Form(...),
    thumbs_up: int = Form(...),
    thumbs_down: int = Form(...),
):
    update_feedback(session_id=session_id, message_id=message_id, up=thumbs_up, down=thumbs_down)
    return {"status": "Feedback recorded successfully"}

# ===================== /test (mock) =====================
@app.post("/test")
async def test_endpoint(message: str = Form(...), file: UploadFile = File(None)):
    if not message:
        raise HTTPException(status_code=400, detail="No message provided")

    return {
        "mermaidCode": "flowchart LR\nA-->B",
        "endMessage": "this is a response to " + message,
        "messages": [
            {"name": "Supervisor", "text": "Mensaje del supervisor"},
            {"name": "researcher", "text": "Mensaje del investigador"},
        ],
    }
