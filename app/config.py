from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Anthropic
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-6"

    # OpenAI-compatible (vLLM local models)
    openai_base_url: str = ""
    openai_api_key: str = "dummy"

    # Google Gemini (optional)
    google_api_key: str = ""

    # Database (PostgreSQL — shared with n8n)
    database_url: str

    # Teamwork (optional)
    teamwork_api_key: str = ""
    teamwork_domain: str = ""  # e.g. "yourcompany.teamwork.com"

    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
