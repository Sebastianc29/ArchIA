# back/src/quoting.py
from pathlib import Path
import re
from typing import List, Dict, Any

def _safe_page(meta: Dict[str, Any]) -> int | None:
    p = meta.get("page")
    if isinstance(p, int):
        return p + 1 if p >= 0 else p
    p2 = meta.get("page_number")
    return p2 if isinstance(p2, int) else None

def pack_quotes(docs, max_quotes: int = 4, max_chars: int = 600) -> List[Dict[str, Any]]:
    quotes = []
    for d in docs[:max_quotes]:
        meta = d.metadata or {}
        source = meta.get("source_path") or meta.get("source") or ""
        title = meta.get("title") or (Path(source).stem if source else "doc")
        page = _safe_page(meta)
        txt = (d.page_content or "").strip()
        # Limpieza ligera de encabezados numéricos (opcional)
        txt = re.sub(r"^\s*\d{1,4}\s*$", "", txt, flags=re.MULTILINE).strip()
        if not txt:
            continue
        if len(txt) > max_chars:
            txt = txt[:max_chars].rsplit(" ", 1)[0] + "…"
        quotes.append({"title": title, "page": page, "source": source, "text": txt})
    return quotes

def render_quotes_md(quotes: List[Dict[str, Any]]) -> str:
    if not quotes:
        return ""
    lines = ["\n## Citas relevantes\n"]
    for q in quotes:
        head = f"**{q['title']}**" + (f" — p. {q['page']}" if q.get("page") is not None else "")
        src  = f" ({q['source']})" if q.get("source") else ""
        lines.append(f"> {q['text']}\n\n— {head}{src}\n")
    return "\n".join(lines)
