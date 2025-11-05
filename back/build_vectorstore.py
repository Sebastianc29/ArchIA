# back/build_vectorstore.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv, find_dotenv
from src.rag_agent import _embeddings as _embeddings_factory
try:
    from langchain_chroma import Chroma
except Exception:  # pragma: no cover
    from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================== Paths / Config ==================
load_dotenv(find_dotenv())
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
PERSIST_DIR = Path(os.environ.get("CHROMA_DIR", str(BASE_DIR / "chroma_db")))

DOCS_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "arquia"
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Libros permitidos (título lógico -> patrones para encontrar el PDF)
ALLOWED_BOOKS: Dict[str, List[str]] = {
    "Software Architecture in Practice (3e)": [
        "Software Architecture in practice.pdf",
        "Software_Architecture_in_Practice",
        "Architecture in Practice",
    ],
    "Evaluating Software Architectures": [
        "Software architectures evaluation.pdf",
        "Evaluating Software Architectures",
        "Architecture Evaluation",
    ],
}


def _match_pdf(files: List[Path], patterns: List[str]) -> Path | None:
    """
    Busca el primer archivo cuyo nombre contenga alguno de los patrones (case-insensitive).
    """
    low_patterns = [p.lower() for p in patterns]
    for f in files:
        fname = f.name.lower()
        if any(p in fname for p in low_patterns):
            return f
    return None


def _load_docs() -> List:
    """
    Carga en memoria las páginas de los PDFs permitidos, con metadatos uniformes.
    """
    all_pdfs = sorted(list(DOCS_DIR.glob("*.pdf")))
    if not all_pdfs:
        print(f"[build] No se encontraron PDFs en {DOCS_DIR}.")
        return []

    docs = []
    for source_title, patterns in ALLOWED_BOOKS.items():
        fpath = _match_pdf(all_pdfs, patterns)
        if not fpath:
            print(f"[build] (Aviso) No se encontró PDF para: {source_title}. "
                  f"Coloca un archivo que contenga uno de: {patterns}")
            continue

        print(f"[build] Cargando: {source_title}  <-- {fpath.name}")
        loader = PyPDFLoader(str(fpath))
        pages = loader.load()
        # normaliza metadatos
        for d in pages:
            md = d.metadata or {}
            d.metadata = {
                "title": Path(md.get("source") or fpath).name,   # nombre del archivo
                "source_title": source_title,                    # título lógico del libro (para filtrar)
                "source_path": str(fpath),                       # ruta absoluta
                "page": md.get("page", md.get("page_number")),   # número de página (int)
                "page_label": md.get("page_label"),              # etiqueta (si existe)
            }
        docs.extend(pages)

    return docs


def _split_docs(docs: List):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def main():
    print("[build] docs_dir          =", DOCS_DIR)
    print("[build] persist_directory =", PERSIST_DIR)

    docs = _load_docs()
    if not docs:
        print("[build] No hay documentos válidos. Copia los PDFs a /back/docs y reintenta.")
        raise SystemExit(1)

    chunks = _split_docs(docs)
    print(f"[build] Total chunks: {len(chunks)}")

    # Embeddings y vectorstore
    emb = _embeddings_factory()  # usa Azure/OpenAI según .env
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
    )
    vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=emb,
    persist_directory=str(PERSIST_DIR),
    collection_name=COLLECTION_NAME,
)

# Persistencia:
# - langchain-chroma: ya quedó persistido automáticamente
# - langchain_community: requiere .persist()
    if hasattr(vectordb, "persist"):
        try:
            vectordb.persist()
        except Exception:
            pass

    print("[build] ¡Vector store construido y persistido!")

    # Resumen (fuentes únicas)
    titles = []
    seen = set()
    for d in chunks[:1000]:  # muestrario
        t = (d.metadata or {}).get("title")
        if t and t not in seen:
            seen.add(t)
            titles.append(t)
    print("[build] Fuentes (muestra):")
    for t in titles[:10]:
        print("   -", t)

    print(f"[build] Listo. BD en: {PERSIST_DIR}")


if __name__ == "__main__":
    main()
