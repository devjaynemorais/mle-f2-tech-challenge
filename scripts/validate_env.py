"""Validate that all required packages and environment variables are present."""

import importlib
import sys


REQUIRED_PACKAGES = [
    "torch",
    "sklearn",
    "mlflow",
    "dvc",
    "pydantic_settings",
    "yaml",
    "pandas",
    "numpy",
]

REQUIRED_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
]


def check_packages() -> list[str]:
    """Return list of packages that cannot be imported."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def check_env_vars() -> list[str]:
    """Return list of unset required environment variables."""
    import os

    return [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]


def main() -> None:
    """Run all environment checks and exit with code 1 on failure."""
    ok = True

    missing_pkgs = check_packages()
    if missing_pkgs:
        print(f"[FAIL] Missing packages: {missing_pkgs}")
        ok = False
    else:
        print("[OK] All required packages are installed.")

    missing_vars = check_env_vars()
    if missing_vars:
        print(f"[WARN] Env vars not set: {missing_vars} (will use defaults from settings.py)")
    else:
        print("[OK] All required env vars are set.")

    if not ok:
        sys.exit(1)

    print("\nEnvironment validation passed.")


if __name__ == "__main__":
    main()
