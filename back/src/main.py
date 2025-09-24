from typing import Optional
from pathlib import Path
import os, re, sqlite3, base64

from fastapi import UploadFile, File, Form, HTTPException, Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routers.diagram import router as diagram_router

from langchain_core.messages import HumanMessage
from src.graph import graph
from src.rag_agent import create_or_load_vectorstore
from src.memory import init as memory_init, get as memory_get, set_kv as memory_set
from src.clients.kroki_client import render_kroki_sync  # <- para el fallback

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

app.include_router(diagram_router)

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

# ====== Fallback builder (solo cuando lo piden o falta) ======
def _build_component_puml_from_text(text: str, title_hint: str = "") -> str:
    t = (text or "").lower()
    def has(*kw): return any(k in t for k in kw)

    nodes = ['actor User as USER', 'component "Backend FastAPI" as API']
    edges = ['USER --> API : HTTP(S) request']

    if has("front", "react", "vite", "ui", "web"):
        nodes.append('component "Frontend (React/Vite)" as FE')
        edges += ['USER --> FE : Browser', 'FE --> API : REST/JSON']

    if has("gateway", "nginx", "ingress", "api gateway"):
        nodes.append('component "API Gateway" as GATE')
        edges += ['FE --> GATE', 'GATE --> API']

    if has("auth", "oauth", "jwt", "keycloak"):
        nodes.append('component "Auth Service" as AUTH')
        edges.append('API --> AUTH : validate token')

    if has("db", "database", "postgres", "mysql", "mongodb"):
        nodes.append('database "DB" as DB')
        edges.append('API --> DB : SQL/NoSQL')

    if has("cache", "redis"):
        nodes.append('component "Cache (Redis)" as CACHE')
        edges.append('API --> CACHE')

    if has("queue", "broker", "kafka", "rabbit"):
        nodes.append('component "Message Broker" as MQ')
        edges.append('API --> MQ : events')

    if has("rag", "vector", "embedding", "chroma", "pdf"):
        nodes.append('component "Vector Store (Chroma)" as VS')
        nodes.append('component "Docs Storage (PDFs)" as DOCS')
        edges += ['API --> VS : search', 'API --> DOCS : load']

    if has("llm", "openai", "azure openai", "model"):
        nodes.append('component "LLM Provider" as LLM')
        edges.append('API --> LLM : inference')

    if has("kroki", "plantuml", "c4"):
        nodes.append('component "Kroki" as KROKI')
        edges.append('API --> KROKI : render diagram')

    if len(nodes) <= 2:
        nodes += ['database "DB" as DB', 'component "Kroki" as KROKI']
        edges += ['API --> DB', 'API --> KROKI']

    body = "\n".join(nodes + [""] + edges)
    title = (title_hint or "Component Diagram").strip()[:60]
    return "@startuml\n!pragma teoz true\n" + f"title {title}\n\n{body}\n@enduml\n"

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

    # ID incremental para feedback por mensaje
    message_id = get_next_message_id(session_id)

    # Hilo único por turno (si quieres memoria full por sesión, usa thread_id = session_id)
    thread_id = f"{session_id}:{message_id}"

    # --- Manejo de imágenes (opcionales) ---
    image_path1 = ""
    if image1 and image1.filename:
        dest1 = IMAGES_DIR / image1.filename
        with open(dest1, "wb") as f:
            f.write(await image1.read())
        image_path1 = str(dest1)

    image_path2 = ""
    if image2 and image2.filename:
        dest2 = IMAGES_DIR / image2.filename
        with open(dest2, "wb") as f:
            f.write(await image2.read())
        image_path2 = str(dest2)

    # --- Turno actual como HumanMessage(s) ---
    turn_messages = [HumanMessage(content=message)]
    if image_path1:
        turn_messages.append(HumanMessage(content=f"[image_path_1] {image_path1}"))
    if image_path2:
        turn_messages.append(HumanMessage(content=f"[image_path_2] {image_path2}"))

    # --- Memoria previa ---
    last_topic = memory_get(user_id, "topic", "")
    asr_prev   = memory_get(user_id, "current_asr", "")
    asr_notes  = memory_get(user_id, "asr_notes", "")
    memory_text = f"Tema previo: {last_topic}. ASR previas: {asr_notes}".strip() or "N/A"

    # --- ASR pegado por el usuario (si lo hay) ---
    asr_in_msg = _extract_asr_from_message(message)
    if asr_in_msg:
        memory_set(user_id, "current_asr", asr_in_msg)
    made_asr = _looks_like_make_asr(message)

    # --- Config del grafo ---
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
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

    # --- Limpieza parcial del estado (sin borrar historial) ---
    try:
        graph.update_state(config, {"values": {
            "endMessage": "",
            "mermaidCode": "",
            "diagram": None,
            "hasVisitedDiagram": False,
            "turn_messages": [],
            "current_asr": memory_get(user_id, "current_asr", ""),
        }})
    except Exception:
        pass  # no crítico

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
                "endMessage": "",
                "mermaidCode": "",
                "turn_messages": [],
                "retrieved_docs": [],
                "memory_text": memory_text,
                "suggestions": [],
                "language": user_lang,
                "intent": "general",
                "force_rag": force_rag,
                "topic_hint": topic_hint,
                "current_asr": memory_get(user_id, "current_asr", ""),
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

    # ===================== DIAGRAMA =====================
    # Usa el diagrama que venga del agente. SOLO usa fallback si:
    #   (1) el usuario pidió “diagram ... of that ASR”, o
    #   (2) NO llegó ningún diagrama (missing).
    diagram_obj = result.get("diagram") or {}
    new_b64 = (diagram_obj.get("svg_b64") or "").strip()
    prev_b64 = memory_get(user_id, "last_svg_b64", "").strip()

    wants_that_asr_diagram = _wants_diagram_of_that_asr(message)
    missing = (not diagram_obj) or (not diagram_obj.get("data_uri"))

    if wants_that_asr_diagram or missing:
        current_asr = memory_get(user_id, "current_asr", "")
        combo_text = (f"{message}\n\n{current_asr}").strip()
        puml = _build_component_puml_from_text(combo_text, title_hint=f"ASR Diagram [{message_id}]")
        ok, payload, err = render_kroki_sync("plantuml", puml, output_format="svg")
        if ok and payload:
            svg_b64 = base64.b64encode(payload).decode("ascii")
            diagram_obj = {
                "ok": True,
                "diagram_type": "plantuml",
                "format": "svg",
                "svg_b64": svg_b64,
                "data_uri": f"data:image/svg+xml;base64,{svg_b64}",
                "message": None,
                "source_echo": puml,
            }
            new_b64 = svg_b64
        else:
            diagram_obj = {
                "ok": False,
                "diagram_type": "plantuml",
                "format": "svg",
                "svg_b64": None,
                "data_uri": None,
                "message": err or "Kroki render error (fallback)",
                "source_echo": puml,
            }
            new_b64 = ""

    # Guarda el último solo si hay algo nuevo
    if new_b64:
        memory_set(user_id, "last_svg_b64", new_b64)

    # --- Payload al front (no pisamos suggestions si las necesitas) ---
    clean_payload = {
        "endMessage": end_msg,
        "mermaidCode": result.get("mermaidCode", ""),
        "diagram": diagram_obj,                # <- usa diagram.data_uri o svg_b64 en el front
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
