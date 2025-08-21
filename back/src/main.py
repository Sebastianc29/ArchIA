from typing import Optional
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import sqlite3

from src.graph import graph

# ===================== Rutas robustas (independientes del cwd) =====================
BACK_DIR = Path(__file__).resolve().parent.parent          # .../back/
IMAGES_DIR = BACK_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_DIR = BACK_DIR / "feedback_db"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DB_PATH = FEEDBACK_DIR / "feedback.db"

# ===================== DB de feedback: init y conexión única ======================
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
    cur.execute(
        "SELECT MAX(message_id) FROM message_feedback WHERE session_id = ?",
        (session_id,),
    )
    row = cur.fetchone()
    # Si no hay mensajes aún para esa sesión, arrancamos en 1
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

# ===================== FastAPI + CORS =============================================
app = FastAPI(title="ArquIA API")

# Permite front local. Agrega 127.0.0.1 por si abres así el front.
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

@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ===================== Endpoints ===================================================

@app.post("/message")
async def message(
    message: str = Form(...),
    session_id: str = Form(...),
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
):
    if not message:
        raise HTTPException(status_code=400, detail="No message provided")
    if not session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")

    # ID incremental por sesión
    message_id = get_next_message_id(session_id)

    # Guardar imágenes (si llegan)
    image_path1 = ""
    if image1 and image1.filename:
        dest1 = IMAGES_DIR / image1.filename
        with open(dest1, "wb") as f:
            f.write(await image1.read())
        image_path1 = str(dest1)  # ruta absoluta para uso interno en el back

    image_path2 = ""
    if image2 and image2.filename:
        dest2 = IMAGES_DIR / image2.filename
        with open(dest2, "wb") as f:
            f.write(await image2.read())
        image_path2 = str(dest2)

    # Mensajes para el agente
    messageList = [{"role": "user", "content": message}]
    if image_path1:
        messageList.append({"role": "user", "content": "this is the image path: " + image_path1})
    if image_path2:
        messageList.append({"role": "user", "content": "this is the second image path: " + image_path2})

    # Thread por sesión
    config = {"configurable": {"thread_id": str(session_id)}}

    # Invocar el grafo
    try:
        result = graph.invoke(
            {
                "messages": messageList,
                "userQuestion": message,
                "localQuestion": "",
                "hasVisitedInvestigator": False,
                "hasVisitedCreator": False,
                "hasVisitedEvaluator": False,
                "hasVisitedASR": False,   # <-- corregido (antes: hasVisitedDiagrams)
                "nextNode": "supervisor",
                "imagePath1": image_path1,
                "imagePath2": image_path2,
                "endMessage": "",
                "mermaidCode": "",
            },
            config,
        )
    except Exception as e:
        # Log simple y error 500 legible al front
        raise HTTPException(status_code=500, detail=f"Graph error: {e}")

    # Registrar feedback base para ese mensaje (0/0)
    upsert_feedback(session_id=session_id, message_id=message_id, up=0, down=0)

    # Respuesta incluyendo IDs
    result["message_id"] = message_id
    result["session_id"] = session_id
    return result

@app.post("/feedback")
async def feedback(
    session_id: str = Form(...),
    message_id: int = Form(...),
    thumbs_up: int = Form(...),
    thumbs_down: int = Form(...),
):
    update_feedback(session_id=session_id, message_id=message_id, up=thumbs_up, down=thumbs_down)
    return {"status": "Feedback recorded successfully"}

# Endpoint de prueba (mock) opcional para el front
@app.post("/test")
async def test_endpoint(message: str = Form(...), file: UploadFile = File(None)):
    if not message:
        raise HTTPException(status_code=400, detail="No message provided")

    user_input = message
    return {
        "mermaidCode": "classDiagram\nA --> B",
        "endMessage": "this is a response to " + user_input,
        "messages": [
            {"name": "Supervisor", "text": "Mensaje del supervisor"},
            {"name": "researcher", "text": "Mensaje del investigador"},
        ],
    }
