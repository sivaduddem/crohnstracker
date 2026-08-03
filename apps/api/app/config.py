from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_mode: str = "demo"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    demo_data_dir: Path = REPO_ROOT / "data" / "demo"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    novelty_distance_threshold: float = 0.55
    low_confidence_threshold: float = 0.55

    @property
    def mode(self) -> str:
        return self.app_mode.lower().strip()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
