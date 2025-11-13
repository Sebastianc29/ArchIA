# src/utils/json_helpers.py
from __future__ import annotations
import json, re
from typing import Any, Iterable, List, Dict, Tuple

_JSON_FENCE_RE = re.compile(r"```(?:jsonc?|JSONC?|Json|JSON)?\s*([\s\S]*?)\s*```", re.M)
_SLASH_SLASH_RE = re.compile(r"^\s*//.*?$", re.M)
_SLASH_STAR_RE = re.compile(r"/\*.*?\*/", re.S)

def _strip_code_fences(text: str) -> Tuple[str, bool]:
    """
    Si hay un bloque ```json ...```, devuelve su contenido y True.
    Si no, devuelve el texto original y False.
    """
    m = _JSON_FENCE_RE.search(text or "")
    if not m:
        return text, False
    return m.group(1), True

def _sanitize_jsonc(s: str) -> str:
    """
    - Quita comentarios // y /* ... */
    - Quita comas colgantes
    - Normaliza comillas “ ” a "
    - No intenta convertir a sintaxis Python (nada de True/False/None)
    """
    s = s or ""
    s = _SLASH_STAR_RE.sub("", s)
    s = _SLASH_SLASH_RE.sub("", s)
    # comas colgantes antes de ] o }
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # comillas tipográficas → dobles
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'")
    return s.strip()

def _first_braced_fragment(text: str) -> str | None:
    """
    Busca el primer {...} o [...] (no balancea a nivel de parser, pero funciona bien para casos típicos).
    Prefiere arrays; si no hay, toma objeto.
    """
    if not text:
        return None
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        return m.group(0)
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        return m.group(0)
    return None

def extract_json_array(text: str):
    """
    Devuelve una lista (array JSON) si la encuentra en `text`.
    Busca primero un fence ```json ... ```, luego un fence ``` ... ```,
    y por último un bloque crudo desde el primer '[' hasta el último ']'.
    Aplica pequeñas reparaciones seguras (coma decimal, comas colgantes).
    """
    if not text:
        return []

    s = str(text)

    # 1) ```json ... ```
    m = re.search(r"```json\s*(.*?)\s*```", s, flags=re.I | re.S)
    if m:
        blob = m.group(1).strip()
    else:
        # 2) ``` ... ``` (sin etiqueta json) que contenga un array
        m = re.search(r"```\s*(\[\s*{.*?}\s*\])\s*```", s, flags=re.S)
        if m:
            blob = m.group(1).strip()
        else:
            # 3) bloque crudo [ ... ]
            i, j = s.find("["), s.rfind("]")
            if i == -1 or j == -1 or j <= i:
                return []  # <- SALIDA LIMPIA: no hay JSON visible
            blob = s[i:j+1]

    # --- Reparaciones tolerantes (solo sobre `blob`) ---
    blob = blob.replace("\r", "")
    # comillas "inteligentes" → ASCII
    blob = blob.replace("“", '"').replace("”", '"').replace("’", "'")
    # 0,82 -> 0.82 en valores numéricos después de ':'
    blob = re.sub(r'(:\s*)(-?\d+),(\d+)(\s*[,\}])', r'\1\2.\3\4', blob)
    # quitar comas colgantes: { ... ,} o [ ... ,]
    blob = re.sub(r",(\s*[\}\]])", r"\1", blob)

    try:
        data = json.loads(blob)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def strip_first_json_fence(text: str) -> str:
    """
    Elimina SOLO el primer bloque ```json ...``` del texto y deja el resto (markdown explicativo).
    Si no hay bloque, retorna el texto igual.
    """
    if not text:
        return ""
    return _JSON_FENCE_RE.sub("", text, count=1).strip()

def _coerce_prob(v: Any) -> float:
    """
    Convierte porcentajes '82%' o strings a float [0,1]. Si no puede, devuelve 0.0
    """
    try:
        if isinstance(v, str) and v.strip().endswith("%"):
            return max(0.0, min(1.0, float(v.strip().strip("%")) / 100.0))
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return 0.0

def normalize_tactics_json(items: Iterable[Dict[str, Any]], *, top_n: int = 3) -> List[Dict[str, Any]]:
    """
    - Toma lista de tácticas (dicts) y:
      * se queda con dicts válidos
      * normaliza success_probability a [0,1]
      * ordena desc por success_probability
      * recorta a top_n
      * asegura rank único 1..n
      * completa campos mínimos vacíos si faltan
    """
    cleaned = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        prob = _coerce_prob(
            it.get("success_probability") or it.get("probability") or it.get("score")
        )
        name = (it.get("name") or "").strip()
        cleaned.append({
            "name": name,
            "purpose": it.get("purpose", "").strip() if isinstance(it.get("purpose"), str) else "",
            "rationale": it.get("rationale", "").strip() if isinstance(it.get("rationale"), str) else "",
            "risks": list(it.get("risks") or []),
            "tradeoffs": list(it.get("tradeoffs") or []),
            "categories": list(it.get("categories") or []),
            "traces_to_asr": it.get("traces_to_asr", "").strip() if isinstance(it.get("traces_to_asr"), str) else "",
            "expected_effect": it.get("expected_effect", "").strip() if isinstance(it.get("expected_effect"), str) else "",
            "success_probability": prob
        })

    cleaned.sort(key=lambda d: d["success_probability"], reverse=True)
    top = cleaned[:max(1, top_n)]
    for i, it in enumerate(top, 1):
        it["rank"] = i
    return top

def build_json_from_markdown(md: str, *, top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Fallback sencillo: detecta líneas 'Name — XXX' o 'Name - XXX' y arma objetos mínimos.
    """
    if not md:
        return []
    names = re.findall(r"(?im)^\s*Name\s*[—-]\s*(.+)$", md)
    base_probs = [0.82, 0.75, 0.68]
    out = []
    for i, raw in enumerate(names[:top_n], 1):
        n = raw.strip().rstrip(".")
        out.append({
            "name": n,
            "purpose": "",
            "rationale": "",
            "risks": [],
            "tradeoffs": [],
            "categories": [],
            "traces_to_asr": "",
            "expected_effect": "",
            "success_probability": base_probs[i-1],
            "rank": i
        })
    return out
