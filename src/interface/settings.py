from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # URL сервиса
    ANALYZER_BASE_URL: str = "http://localhost:8000"
    DATABASE_URL: str = "sqlite:///./reviews.db"

settings = Settings()