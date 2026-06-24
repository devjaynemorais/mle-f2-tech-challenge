"""Application settings loaded from environment variables / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for all pipeline stages and services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "recommendation_system"

    # Data paths
    data_raw_path: str = "data/raw"
    data_interim_path: str = "data/interim"
    data_processed_path: str = "data/processed"

    # Model artifacts
    model_artifacts_path: str = "models/artifacts"
    model_type: str = "mlp"

    # Reproducibility
    random_state: int = 42
    test_size: float = 0.2

    # API serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # AWS — optional, only for DVC remote S3 / cloud deploy
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_default_region: str = "us-east-1"


settings = Settings()
