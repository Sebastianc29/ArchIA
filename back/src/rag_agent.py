# src/rag_agent.py
from __future__ import annotations

import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

# Usa el paquete nuevo si está instalado; si no, cae al community.
try:
    from langchain_chroma import Chroma
except Exception:  # pragma: no cover
    from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings

# ================== Paths / Config ==================

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # .../back
DEFAULT_CHROMA_DIR = str((PROJECT_ROOT / "chroma_db").resolve())

# Modelo de embeddings por env, con default razonable
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Nombre de colección (consistente con el build)
COLLECTION_NAME = "arquia"

# Singleton del vectorstore
_VDB: Chroma | None = None

def _embeddings():
    """Crea la función de embeddings según env. Usa Azure si hay vars AZURE_*."""
    az_key = os.getenv("AZURE_OPENAI_API_KEY")
    az_ep  = os.getenv("AZURE_OPENAI_ENDPOINT")
    if az_key and az_ep:
        # ---- Azure OpenAI ----
        # Debes tener creado un deployment de embeddings en Azure y poner su nombre:
        az_embed_deploy = (
            os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_EMBED_MODEL")
        )
        if not az_embed_deploy:
            raise RuntimeError(
                "Falta AZURE_OPENAI_EMBED_DEPLOYMENT en .env (deployment de embeddings en Azure)."
            )
        az_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        return AzureOpenAIEmbeddings(
            azure_deployment=az_embed_deploy,
            api_key=az_key,
            azure_endpoint=az_ep,
            api_version=az_api_version,
        )
    else:
        # ---- OpenAI clásico ----
        # Requiere OPENAI_API_KEY y (opcional) OPENAI_BASE_URL si usas proxy
        model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
        return OpenAIEmbeddings(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")  # opcional
        )

def create_or_load_vectorstore() -> Chroma:
    """
    Carga la BD de Chroma ya persistida (construida por build_vectorstore.py).
    Si la carpeta está vacía, la instancia se crea igualmente (sin data).
    """
    global _VDB
    if _VDB is not None:
        return _VDB

    persist_directory = os.environ.get("CHROMA_DIR", DEFAULT_CHROMA_DIR)
    print(f"[RAG] persist_directory = {persist_directory}")

    # Solo cargar (el build se hace con back/build_vectorstore.py)
    _VDB = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings(),
        persist_directory=persist_directory,
    )
    # Nota: si no hay datos aún, _VDB está “vacío” pero funcional.
    return _VDB


# src/rag_agent.py  (reemplaza tu get_retriever por este)
def get_retriever(title: str | list[str] | None = None, k: int = 6):
    """
    Devuelve un retriever del vector store.
    - Si `title` es string: filtra por igualdad exacta en metadata.title
    - Si `title` es lista: usa $in para cualquiera
    """
    vectorstore = create_or_load_vectorstore()
    if isinstance(title, list) and title:
        return vectorstore.as_retriever(
            search_kwargs={"k": k, "filter": {"title": {"$in": title}}}
        )
    if isinstance(title, str) and title:
        return vectorstore.as_retriever(
            search_kwargs={"k": k, "filter": {"title": {"$eq": title}}}
        )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def rebuild_vectorstore():
    """
    Helper opcional: si quieres reconstruir en caliente, borra el dir
    y vuelve a cargar (pero el pipeline recomendado es usar build_vectorstore.py).
    """
    import shutil
    global _VDB
    persist_directory = os.environ.get("CHROMA_DIR", DEFAULT_CHROMA_DIR)
    if os.path.isdir(persist_directory):
        print(f"[RAG] Removing existing Chroma DB at {persist_directory}")
        shutil.rmtree(persist_directory, ignore_errors=True)
    _VDB = None
    return create_or_load_vectorstore()
