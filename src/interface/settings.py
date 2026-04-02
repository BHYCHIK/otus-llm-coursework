from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="UI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    ANALYZER_BASE_URL: str = ""
    DATABASE_URL: str = ""

settings = Settings()
print(settings)