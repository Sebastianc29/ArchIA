# src/memory.py
import sqlite3, os, json
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "state_db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "memory.db"

def _conn():
    return sqlite3.connect(str(DB_PATH))

def init():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS memory (
            user_id TEXT, key TEXT, value TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, key)
        )""")

def set_kv(user_id: str, key: str, value: str):
    with _conn() as c:
        c.execute("""INSERT INTO memory(user_id, key, value) VALUES(?,?,?)
                     ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP""",
                  (user_id, key, value))

def get(user_id: str, key: str, default: str = "") -> str:
    with _conn() as c:
        row = c.execute("SELECT value FROM memory WHERE user_id=? AND key=?", (user_id, key)).fetchone()
        return row[0] if row else default

###
### Nueva memoria persistente para el flujo ADD 3.0
###

# Clave unica para guardar todo el contexto de arquirectura
ARCH_FLOW_KEY = "arch_flow"

def empty_arch_flow() -> dict:
    """
    Estado minimo que queremos recordar entre turnos
    stage:
         ASR: ya tenemos un ASR válido
         STYLE: ya hay un estilo arquitectonico elegido
         Tactics: ya se discutió las técnicas
         Deployment: ya hay diagrama de despliegue
    """
    return {
        "stage":"",     
        "quality_attribute": "", # Eje: "availability", "latency"
        "add_context":"", #dominio / driver de negocio
        "current_asr":"", # ASR oficial
        "style":"", # unico estilo actualmente elegido
        "tactics": [], #Lista de tácticas aceptadas
        "deployment_diagram_puml":"", #PlantUML del despliegue final
        "deployment_diagram_svg_b64":"", #SVG base 64 del despliegue final
    }

def load_arch_flow(user_id: str) -> dict:
    """
    Devuelve el estado ADD 3.0 para este usuario/sesión
    Siempre retorna todas las llaves esperadas.
    """
    raw = get(user_id, ARCH_FLOW_KEY, "")
    if not raw:
        return empty_arch_flow()
    try:
        data = json.loads(raw)
    except Exception:
        data = {}
    base = empty_arch_flow()
    base.update(data or {})
    return base

def save_arch_flow(user_id: str, flow: dict):
    """
    Guarda el estado ADD 3.0 actualizado
    """
    base = empty_arch_flow()
    base.update(flow or {})
    set_kv(user_id, ARCH_FLOW_KEY, json.dumps(base))