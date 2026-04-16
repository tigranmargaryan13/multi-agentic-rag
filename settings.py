from typing import Literal, Optional
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class BaseLLMSettings(BaseSettings):
    RETRIEVE_DOCS_NUMBER: int = 4
    LLM_MODEL_NAME: str = "gpt-4.1-mini"
    LLM_TEMPERATURE: Optional[float] = 0.0

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHUNK_SIZE: int = 2048
    CHUNK_OVERLAP: int = 50

    OPENAI_API_KEY: Optional[str]

    FAISS_INDEX_DIR: str = "./faiss_index"
    DOCUMENTS_BASE_DIR: str = "./documents"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",
    )
