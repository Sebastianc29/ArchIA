from typing import Optional
from pathlib import Path
import os, re, sqlite3

from fastapi import UploadFile, File, Form, HTTPException, Request, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.graph import graph
from src.rag_agent import create_or_load_vectorstore
from src.memory import init as memory_init, get as memory_get, set_kv as memory_set

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

# Una sola instancia de FastAPI (evita duplicados)
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

# Helpers
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

    # --- Mensajes base para el grafo ---
    messageList = [{"role": "user", "content": message}]
    if image_path1:
        messageList.append({"role": "user", "content": "this is the image path: " + image_path1})
    if image_path2:
        messageList.append({"role": "user", "content": "this is the second image path: " + image_path2})

    # --- Memoria previa para continuidad de conversación ---
    last_topic = memory_get(user_id, "topic", "")
    asr_notes  = memory_get(user_id, "asr_notes", "")
    memory_text = f"Tema previo: {last_topic}. ASR previas: {asr_notes}".strip() or "N/A"

    # --- Config del hilo LangGraph ---
    config = {"configurable": {"thread_id": str(session_id)}}
    user_lang = detect_lang(message)

    # --- Heurísticas locales para topic_hint y force_rag ---
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

    topic_hint = _extract_topic_from_text(message) or _extract_topic_from_text(last_topic)

    msg_low = message.lower()
    force_rag = (
        _needs_topic_hint(message) or
        bool(re.search(
            r"\b(add|qas|asr|tactic|táctica|latenc|scalab|throughput|rendim|availability|disponib|diagrama|diagram)\b",
            msg_low
        ))
    )

    # --- Limpia estado residual del hilo (por si quedó algo del turno anterior) ---
    try:
        state = graph.get_state(config)
        if state:
            graph.update_state(config, {"endMessage": "", "mermaidCode": ""})
    except Exception:
        # no es crítico si falla
        pass

    # --- Invocación del grafo ---
    try:
        result = graph.invoke(
            {
                "messages": messageList,
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
                "topic_hint": topic_hint,  # si tu GraphState no lo tiene, puedes quitar esta línea
            },
            config,
        )
    except Exception as e:
        # Imprime traza completa en consola para depurar rápidamente
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph error: {e}")

    # --- Feedback inicial para este mensaje ---
    upsert_feedback(session_id=session_id, message_id=message_id, up=0, down=0)

    # --- Actualiza memoria simple por palabras clave ---
    low = message.lower()
    if "latencia" in low:
        memory_set(user_id, "topic", "latencia")
    elif "escalabilidad" in low:
        memory_set(user_id, "topic", "escalabilidad")
    if "asr" in low:
        memory_set(user_id, "asr_notes", message)

    # --- Payload del turno actual (lo que consume el front) ---
    clean_payload = {
        "endMessage": result.get("endMessage", ""),
        "mermaidCode": result.get("mermaidCode", ""),
        "messages": result.get("turn_messages", []),  # solo los internos de este turno
        "session_id": session_id,
        "message_id": message_id,
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
