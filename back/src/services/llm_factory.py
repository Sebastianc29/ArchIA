# src/services/llm_factory.py
from __future__ import annotations

import os
import logging
from typing import Optional, Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI

# ChatOllama es opcional: solo si usas provider=ollama
try:
    from langchain_community.chat_models import ChatOllama  # type: ignore
except Exception:  # pragma: no cover
    ChatOllama = None  # type: ignore

log = logging.getLogger(__name__)


# =============== Utilidades env =================
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Obtiene variable de entorno limpia (sin espacios)."""
    v = os.getenv(name, default)
    if v is not None and isinstance(v, str):
        v = v.strip()
    return v or default


def _auto_provider() -> Literal["azure", "openai", "ollama"]:
    """
    Prioridad:
      1) ROS_LG_LLM_PROVIDER (si está definida)
      2) Si hay credenciales/config de Azure → 'azure'
      3) Si hay OPENAI_API_KEY o OPENAI_BASE_URL → 'openai'
      4) Si no, fallback → 'ollama'
    """
    forced = _env("ROS_LG_LLM_PROVIDER")
    if forced:
        return forced.lower()  # type: ignore

    if _env("AZURE_OPENAI_API_KEY") and _env("AZURE_OPENAI_ENDPOINT"):
        return "azure"

    if _env("OPENAI_API_KEY") or _env("OPENAI_BASE_URL"):
        return "openai"

    return "ollama"


# =============== Fábrica principal =================
def get_chat_model(temperature: Optional[float] = 0.0) -> BaseChatModel:
    """
    Devuelve un ChatModel de LangChain según el provider detectado/env.

    ⚠ Azure: Algunos deployments NO aceptan 'temperature' distinto de 1.0.
      - Si temperature == 1.0 → se envía.
      - Si temperature es otro valor (e.g., 0.0) → NO se envía (server usa default=1).
    """
    provider = _auto_provider()
    log.info("llm_factory: provider=%s", provider)

    # ---------------- Azure OpenAI ----------------
    if provider == "azure":
        endpoint = _env("AZURE_OPENAI_ENDPOINT")
        api_key = _env("AZURE_OPENAI_API_KEY")
        api_version = _env("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        # Acepta ambas por compatibilidad:
        deployment = _env("AZURE_OPENAI_CHAT_DEPLOYMENT") or _env("AZURE_OPENAI_DEPLOYMENT")

        if not all([endpoint, api_key, api_version, deployment]):
            raise RuntimeError(
                "Faltan vars de Azure OpenAI: "
                "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_API_VERSION, AZURE_OPENAI_CHAT_DEPLOYMENT."
            )

        kwargs = dict(
            azure_endpoint=endpoint.rstrip("/"),
            azure_deployment=deployment,
            api_version=api_version,
            api_key=api_key,
        )

        # Muchos deployments en Azure no aceptan temperature != 1
        if temperature is not None and abs(float(temperature) - 1.0) < 1e-9:
            kwargs["temperature"] = 1.0
        # Si temperature es 0.0 u otro, NO lo incluimos → evita el 400 'unsupported_value'

        return AzureChatOpenAI(**kwargs)

    # ---------------- OpenAI (no Azure) ----------------
    if provider == "openai":
        model = _env("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        base_url = _env("OPENAI_BASE_URL")  # opcional (proxy/self-hosted)
        api_key = _env("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("Falta OPENAI_API_KEY para provider=openai.")

        kwargs = dict(
            model=model,
            api_key=api_key,
            temperature=float(temperature) if temperature is not None else 0.0,
        )
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")

        return ChatOpenAI(**kwargs)

    # ---------------- Ollama (fallback local) ----------------
    # Requiere langchain-community
    if ChatOllama is None:
        raise RuntimeError(
            "Provider=ollama pero 'langchain_community.ChatOllama' no está instalado. "
            "Instala: pip install 'langchain-community>=0.3.0'"
        )

    model = _env("OLLAMA_MODEL", "llama3.1:8b-instruct-fp16")
    base_url = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=float(temperature or 0.0),
    )
