# src/services/llm_factory.py
from __future__ import annotations
from typing import Optional, Literal, Any, Dict
import os
from langchain_core.language_models import BaseChatModel

# --------------------------- utilidades ---------------------------

def _auto_provider() -> str:
    """Detecta proveedor automáticamente según variables de entorno."""
    if os.getenv("ROS_LG_LLM_PROVIDER"):
        return os.getenv("ROS_LG_LLM_PROVIDER").lower()
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
        return "openai"
    return "ollama"

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "", "None") else default

def _normalize_alias(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

# Mapas de alias -> variables esperadas en .env
AZURE_ALIASES: Dict[str, Dict[str, str]] = {
    "gpt4omini": {"model_hint": "gpt-4o-mini", "env": "AZURE_OPENAI_DEPLOYMENT_GPT4O_MINI"},
    "gpt4o":     {"model_hint": "gpt-4o",      "env": "AZURE_OPENAI_DEPLOYMENT_GPT4O"},
    "o3mini":    {"model_hint": "o3-mini",     "env": "AZURE_OPENAI_DEPLOYMENT_O3_MINI"},
    "gpt41":     {"model_hint": "gpt-4.1",     "env": "AZURE_OPENAI_DEPLOYMENT_GPT41"},
    "gpt41mini": {"model_hint": "gpt-4.1-mini","env": "AZURE_OPENAI_DEPLOYMENT_GPT41_MINI"},
    "gpt5":     {"model_hint": "gpt-5",      "env": "AZURE_OPENAI_DEPLOYMENT_GPT5"},
    "gpt5mini": {"model_hint": "gpt-5-mini", "env": "AZURE_OPENAI_DEPLOYMENT_GPT5_MINI"},
}

OLLAMA_ALIASES: Dict[str, str] = {
    "llama3.2:3b": "llama3.2:3b",
    "llama3.1:8b": "llama3.1:8b",
    "mistral:7binstruct": "mistral:7b-instruct",
    "qwen2.5:7binstruct": "qwen2.5:7b-instruct",
    "deepseekr1:7b": "deepseek-r1:7b",
    # alias “humanos”:
    "llama33b": "llama3.2:3b",
    "llama318b": "llama3.1:8b",
    "mistral7b": "mistral:7b-instruct",
    "qwen25_7b": "qwen2.5:7b-instruct",
    "deepseekr1_7b": "deepseek-r1:7b",
}

def _resolve_azure_deployment(sel: Optional[str]) -> str:
    """
    Prioridad:
      1) si `sel` tiene alias con ENV mapeada (AZURE_OPENAI_DEPLOYMENT_...), usarla
      2) si `sel` viene, usar `sel` como deployment
      3) AZURE_OPENAI_CHAT_DEPLOYMENT o AZURE_OPENAI_DEPLOYMENT_NAME
    """
    if sel:
        alias = _normalize_alias(sel)
        if alias in AZURE_ALIASES:
            env_name = AZURE_ALIASES[alias]["env"]
            dep = _env(env_name)
            if dep:
                return dep
        return sel

    dep = _env("AZURE_OPENAI_CHAT_DEPLOYMENT") or _env("AZURE_OPENAI_DEPLOYMENT_NAME")
    if dep:
        return dep

    options = ", ".join(sorted({d["model_hint"] for d in AZURE_ALIASES.values()}))
    raise ValueError(
        "Azure: no se pudo resolver el 'deployment name'. "
        "Pasa `model=` con el deployment o exporta AZURE_OPENAI_CHAT_DEPLOYMENT. "
        f"Aliases soportados: {options}"
    )

def _resolve_ollama_model(sel: Optional[str]) -> str:
    if sel:
        k = _normalize_alias(sel)
        return OLLAMA_ALIASES.get(k, sel)
    return _env("OLLAMA_MODEL", "llama3.2:3b")

# --------------------------- fábrica principal ---------------------------

def get_chat_model(
    provider: Optional[Literal["azure", "openai", "ollama"]] = None,
    model: Optional[str] = None,
    agent_type: Optional[Literal["react", "plan_then_act"]] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Fábrica unificada (Azure / OpenAI / Ollama).
    Si `provider` es None, se autodetecta por .env.
    """
    provider = (provider or _auto_provider()).lower()

    temperature = kwargs.pop("temperature", 0.0)
    max_tokens = kwargs.pop("max_tokens", None)
    timeout = kwargs.pop("timeout", None)
    max_retries = kwargs.pop("max_retries", 2)

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI
        deployment = _resolve_azure_deployment(model or _env("ROS_LG_LLM_MODEL"))
        api_version = _env("OPENAI_API_VERSION") or _env("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        endpoint = _env("AZURE_OPENAI_ENDPOINT")
        key = _env("AZURE_OPENAI_API_KEY")
        if not (endpoint and key):
            raise ValueError("Azure: faltan AZURE_OPENAI_ENDPOINT y/o AZURE_OPENAI_API_KEY.")
        return AzureChatOpenAI(
            azure_deployment=deployment,
            openai_api_version=api_version,
            azure_endpoint=endpoint,
            api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        base_url = _env("OPENAI_BASE_URL")
        key = _env("OPENAI_API_KEY", "dummy-key" if base_url else None)
        mdl = model or _env("ROS_LG_LLM_MODEL") or _env("OPENAI_MODEL", "gpt-5-mini")
        if not (key or base_url):
            raise ValueError("OpenAI: faltan OPENAI_API_KEY o OPENAI_BASE_URL (para servidores compatibles).")
        return ChatOpenAI(
            model=mdl,
            api_key=key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except Exception:
            from langchain_community.chat_models import ChatOllama  # type: ignore
        base_url = _env("OLLAMA_BASE_URL", "http://localhost:11434")
        mdl = _resolve_ollama_model(model or _env("ROS_LG_LLM_MODEL"))
        return ChatOllama(
            model=mdl,
            base_url=base_url,
            temperature=temperature,
            num_ctx=kwargs.pop("num_ctx", 4096),
            **kwargs,
        )

    raise ValueError(f"Proveedor desconocido '{provider}'. Usa: azure | openai | ollama.")
