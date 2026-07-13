"""Validate that all required packages and environment variables are present."""

import importlib
import os
import sys
from pathlib import Path

# Running this file by path puts scripts/ on sys.path, not the repo root,
# so `import src` fails. Add the repo root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUIRED_PACKAGES = [
    "torch",
    "sklearn",
    "mlflow",
    "dvc",
    "pydantic_settings",
    "yaml",
    "pandas",
    "numpy",
    "pyarrow",
    "fastapi",
    "uvicorn",
]

REQUIRED_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
]


def check_packages() -> list[str]:
    """Return names of packages that cannot be imported."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def check_env_vars() -> list[str]:
    """Return required env vars that are not set."""
    return [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]


def check_settings() -> str | None:
    """Try to load Settings; return error message on failure."""
    try:
        from src.config.settings import Settings
        Settings()
        return None
    except Exception as exc:
        return str(exc)


def main() -> None:
    """Run all environment checks and exit with code 1 on failure."""
    ok = True

    missing_pkgs = check_packages()
    if missing_pkgs:
        print(f"[FAIL] Missing packages: {missing_pkgs}")
        ok = False
    else:
        print("[OK] All required packages installed.")

    missing_vars = check_env_vars()
    if missing_vars:
        print(f"[WARN] Env vars not set (defaults will be used): {missing_vars}")
    else:
        print("[OK] All required env vars are set.")

    settings_error = check_settings()
    if settings_error:
        print(f"[FAIL] Settings failed to load: {settings_error}")
        ok = False
    else:
        print("[OK] Settings loaded successfully.")

    if not ok:
        sys.exit(1)

    print("\nEnvironment validation passed.")


if __name__ == "__main__":
    main()
