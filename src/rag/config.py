"""Centralized settings loaded from .env / environment."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ollama_host: str = "http://localhost:11434"
    ollama_model_llm: str = "llama3.1:8b"
    ollama_model_embed: str = "nomic-embed-text"

    vllm_base_url: str | None = None
    vllm_model: str = ""
    vllm_api_key: str = "EMPTY"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "rag_default"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"

    phoenix_enabled: bool = False
    phoenix_endpoint: str = "http://localhost:6006"

    data_dir: Path = Path("./data")
    experiments_dir: Path = Path("./experiments")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
