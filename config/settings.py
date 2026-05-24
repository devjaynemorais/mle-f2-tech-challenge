"""Application settings via Pydantic Settings (loaded from .env)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "recommendation_system"

    # Data paths
    data_raw_path: str = "data/raw"
    data_interim_path: str = "data/interim"
    data_processed_path: str = "data/processed"

    # Reproducibility
    random_state: int = 42
    test_size: float = 0.2

    # Model
    model_type: str = "mlp"


settings = Settings()
