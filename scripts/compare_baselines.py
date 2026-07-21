"""Treina e avalia todos os model_type (dummy, logistic, ncf) em sequência.

O DVC (`dvc.yaml`/`make train`) só treina o `model_type` atual do params.yaml.
Este script automatiza a comparação: para cada tipo, sobrescreve
`train.model_type` no params.yaml, roda train + evaluate (mesmos comandos do
Makefile), e ao final restaura o params.yaml e roda promote uma única vez —
o registry já escolhe o melhor run pela métrica configurada em
`registry.metric` (val_auc, por padrão), então promover no final é seguro
mesmo com os baselines fracos também logados no MLflow.

Uso:
    poetry run python -m scripts.compare_baselines
    poetry run python -m scripts.compare_baselines --model-types dummy logistic
    poetry run python -m scripts.compare_baselines --no-promote
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"
TRAIN_METRICS_PATH = ROOT / "metrics" / "train_metrics.json"
EVAL_METRICS_PATH = ROOT / "metrics" / "eval_metrics.json"
COMPARISON_PATH = ROOT / "metrics" / "baseline_comparison.json"

MODEL_TYPE_RE = re.compile(r"^(  model_type:\s*)(\S+)", re.MULTILINE)
ALL_MODEL_TYPES = ["dummy", "logistic", "ncf"]


def _current_model_type(text: str) -> str:
    match = MODEL_TYPE_RE.search(text)
    if not match:
        raise RuntimeError("Não encontrei 'train.model_type' em params.yaml")
    return match.group(2)


def _set_model_type(text: str, model_type: str) -> str:
    new_text, n = MODEL_TYPE_RE.subn(rf"\g<1>{model_type}", text)
    if n != 1:
        raise RuntimeError("Substituição de model_type falhou (0 ou 2+ matches)")
    return new_text


def _run(module: str) -> None:
    print(f"  $ poetry run python -m {module}")
    # No console do Windows a saída padrão costuma vir em cp1252, que não
    # codifica os emojis que o MLflow imprime (ex.: "View run ... at: ...").
    # Sem isso o subprocesso encerra com UnicodeEncodeError mesmo depois de
    # treinar/avaliar com sucesso.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run(
        ["poetry", "run", "python", "-m", module], cwd=ROOT, check=True, env=env
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=ALL_MODEL_TYPES,
        choices=ALL_MODEL_TYPES,
        help="Quais model_type treinar/avaliar, em ordem (default: todos).",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Não roda 'make promote' (src.models.registry) ao final.",
    )
    args = parser.parse_args()

    original_text = PARAMS_PATH.read_text()
    original_model_type = _current_model_type(original_text)
    results: dict[str, dict] = {}

    try:
        for model_type in args.model_types:
            print(f"\n=== model_type={model_type} ===")
            PARAMS_PATH.write_text(_set_model_type(original_text, model_type))
            _run("src.training.trainer")
            _run("src.evaluation.evaluate")
            results[model_type] = {
                "train": _read_json(TRAIN_METRICS_PATH),
                "eval_classification": _read_json(EVAL_METRICS_PATH).get(
                    "classification", {}
                ),
            }
    finally:
        print(f"\nRestaurando params.yaml (model_type={original_model_type})")
        PARAMS_PATH.write_text(original_text)

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nComparação salva em {COMPARISON_PATH}")

    print(f"\n{'model_type':<12} {'val_auc':>10} {'val_ndcg@20':>12} {'test_auc':>10} {'test_f1':>10}")
    for model_type, r in results.items():
        train = r["train"]
        test = r["eval_classification"]
        print(
            f"{model_type:<12} "
            f"{train.get('val_auc', float('nan')):>10.4f} "
            f"{train.get('val_ndcg_at_20', float('nan')):>12.4f} "
            f"{test.get('roc_auc', float('nan')):>10.4f} "
            f"{test.get('f1', float('nan')):>10.4f}"
        )

    if not args.no_promote:
        print("\n=== promote ===")
        _run("src.models.registry")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nFalhou: {exc}", file=sys.stderr)
        sys.exit(exc.returncode)
