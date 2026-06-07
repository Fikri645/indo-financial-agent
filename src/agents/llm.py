"""LLM provider factory — returns the right LangChain chat model based on config.

Mirrors the multi-provider pattern from Philosopher Chat: one env-var switch
(LLM_PROVIDER) selects between Groq, Google AI Studio, or OpenRouter so the
agent runs free on any tier.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src import config


def get_llm(temperature: float | None = None) -> BaseChatModel:
    """Return a configured LangChain chat model for the active provider.

    Parameters
    ----------
    temperature:
        Override the default from ``config.LLM_TEMPERATURE``. Pass 0.0 for
        deterministic structured outputs, higher for creative narrative.
    """
    temp = temperature if temperature is not None else config.LLM_TEMPERATURE
    provider = config.LLM_PROVIDER

    if provider == "groq":
        from langchain_groq import ChatGroq  # local import: optional dependency

        return ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.LLM_MODEL,
            temperature=temp,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=config.GOOGLE_API_KEY,
            model=config.LLM_MODEL,
            temperature=temp,
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model=config.LLM_MODEL,
            temperature=temp,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        "Set LLM_PROVIDER to one of: groq | google | openrouter"
    )
