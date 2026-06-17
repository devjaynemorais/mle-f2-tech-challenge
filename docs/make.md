# Referência do Makefile

Pré-requisito: Python 3.11+, Make, Docker.

---

## Ambiente

| Comando | O que faz |
|---------|-----------|
| `make env` | Instala Poetry 1.8.3 + todas as dependências (prod + dev) em `.venv/` |
| `make install` | Alias de `make env` |

---

## Pipeline Completo

```
make setup
```

Executa em sequência: `validate-env` → `dvc repro`  
Re-executa apenas os estágios cujos inputs mudaram (comportamento do DVC).

### Estágios individuais

| Comando | Módulo executado |
|---------|-----------------|
| `make preprocess` | `src.data.preprocess` |
| `make feature-eng` | `src.features.build_features` |
| `make train` | `src.training.trainer` |
| `make evaluate` | `src.evaluation.evaluate` |

Ordem obrigatória: `preprocess → feature-eng → train → evaluate`

---

## DVC

| Comando | O que faz |
|---------|-----------|
| `make dvc-repro` | Reproduz o pipeline completo via `dvc repro` |
| `make dvc-pull` | Baixa os dados do remote (`dvc-storage/`) |
| `make dvc-push` | Envia dados para o remote |

---

## Qualidade de Código

| Comando | O que faz |
|---------|-----------|
| `make lint` | `ruff check` + verificação de formatação (apenas reporta) |
| `make format` | Corrige problemas de lint e formatação automaticamente |

---

## Testes

| Comando | O que faz |
|---------|-----------|
| `make test` | `pytest tests/ -v` |
| `make test-cov` | pytest com relatório de cobertura em HTML (`htmlcov/`) |

---

## Serviços Locais

| Comando | O que faz |
|---------|-----------|
| `make mlflow` | Sobe MLflow UI em `http://localhost:5000` (SQLite backend) |
| `make api` | Sobe FastAPI em `http://localhost:8000` com hot-reload |
| `make validate-env` | Valida variáveis de ambiente e dependências instaladas |

---

## Docker

| Comando | O que faz |
|---------|-----------|
| `make compose-build` | Builda todas as imagens Docker |
| `make compose-full` | Sobe MLflow + treino + API |
| `make compose-down` | Para e remove os containers |

### Quando usar Docker vs local

| Situação | Usar |
|----------|------|
| Desenvolvimento e debugging | `make api` + `make mlflow` (local) |
| Validar containerização ou CI | `make compose-build && make compose-full` |
| Modelo atualizado no Docker | Sempre `compose-build` antes de `compose-full` |

---

## Fluxo típico de desenvolvimento

```bash
make env              # primeira vez ou após mudar pyproject.toml
make validate-env     # checar se o ambiente está OK
make setup            # rodar pipeline completo
make mlflow           # abrir UI para comparar experimentos
make api              # servir a API localmente
make lint             # antes de commitar
make test             # antes de commitar
```
