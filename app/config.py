"""
Application configuration loaded from environment variables.
"""

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Google / Gemini (either key works for Gemini API)
    google_api_key: str = ""
    gemini_api_key: str = ""

    # Tavily
    tavily_api_key: str = ""

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_user: str = "learnado"
    postgres_password: str = "learnado_secret"
    postgres_db: str = "learnado"

    # SQLAlchemy
    db_echo: bool = False

    # Meta (WhatsApp)
    whatsapp_access_token: str = ""
    meta_verify_token: str = "TESTTOKENFORLEARNADO"
    whatsapp_phone_number_id: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_database_url(self) -> str:
        """Sync URL used by Alembic migrations."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
