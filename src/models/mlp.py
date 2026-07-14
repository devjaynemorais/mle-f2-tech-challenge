"""Modelo principal: NCF (Neural Collaborative Filtering) em PyTorch.

Registrado na Factory com o nome "ncf". Arquitetura (FR-004, D4):
embeddings de usuário e item (+ categoria do item, para cold-start de
conteúdo) concatenados às features contínuas escaladas → MLP → logit.

O ÚLTIMO índice de cada tabela de embedding (``n_users``/``n_items``/
``n_categories``) é reservado para "unknown": ids ausentes do treino são
roteados para ele — o índice 0 é um id real do ``factorize`` (D4).

O modelo é AUTO-CONTIDO (D5): o scaler das features contínuas é ajustado
no ``fit`` e serializado junto no pickle, garantindo que treino, avaliação
e serving apliquem a mesma transformação (FR-007).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.data.dataset import RetailRocketDataset
from src.data.feature_contract import CONT_COLS, CONTRACT_VERSION, ID_COLS
from src.models.base import RecommenderBase
from src.models.factory import ModelFactory

_PREDICT_BATCH = 65_536
_MIN_IMPROVEMENT = 1e-5

# Features de cauda pesada (power-law): log1p antes do scaler, senão a
# informação de popularidade fica espremida em z-scores inúteis.
_LOG_SCALE_COLS = ("frequency", "engagement_score", "view_count")
_LOG_SCALE_IDX = [CONT_COLS.index(c) for c in _LOG_SCALE_COLS]


class _NCFNet(nn.Module):
    """Rede interna: embeddings (user, item, categoria) ⊕ contínuas → logit."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_categories: int,
        embedding_dim: int,
        cat_embedding_dim: int,
        n_cont: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        # +1: último índice reservado para "unknown" (D4).
        self.user_emb = nn.Embedding(n_users + 1, embedding_dim)
        self.item_emb = nn.Embedding(n_items + 1, embedding_dim)
        self.cat_emb = nn.Embedding(n_categories + 1, cat_embedding_dim)
        layers: list[nn.Module] = []
        prev = 2 * embedding_dim + cat_embedding_dim + n_cont
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        cats: torch.Tensor,
        cont: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [self.user_emb(users), self.item_emb(items), self.cat_emb(cats), cont],
            dim=1,
        )
        return self.mlp(x).squeeze(-1)


@ModelFactory.register("ncf")
class NCFRecommender(RecommenderBase):
    """NCF para recomendação com feedback implícito (positivo vs. negativo).

    A entrada de ``fit``/``predict_proba`` é a matriz na ordem do contrato de
    features (ids crus + contínuas cruas); ids e escala são tratados aqui.

    Args:
        n_users: Tamanho do vocabulário de usuários (max user_idx + 1).
        n_items: Tamanho do vocabulário de itens (max item_idx + 1).
        n_categories: Nº de categorias de item conhecidas.
        item_categories: Array (n_items + 1) item_idx → cat_idx; posições sem
            categoria recebem ``n_categories`` (unknown).
        known_users: Máscara bool (n_users) — usuários vistos no TREINO.
        known_items: Máscara bool (n_items) — itens vistos no TREINO.
        embedding_dim: Dimensão dos embeddings de usuário/item.
        cat_embedding_dim: Dimensão do embedding de categoria.
        hidden_dims: Tamanhos das camadas ocultas do MLP.
        dropout: Taxa de dropout.
        unknown_dropout: Fração de ids de treino roteada para o índice
            unknown — treina o embedding de cold-start (D4).
        lr: Taxa de aprendizado (Adam).
        weight_decay: Regularização L2 do Adam.
        epochs: Máximo de épocas.
        batch_size: Tamanho do mini-batch.
        patience: Épocas sem melhora na métrica de validação antes de parar.
        random_state: Semente para reprodutibilidade.
    """

    def __init__(
        self,
        n_users: int = 1,
        n_items: int = 1,
        n_categories: int = 0,
        item_categories: np.ndarray | None = None,
        known_users: np.ndarray | None = None,
        known_items: np.ndarray | None = None,
        embedding_dim: int = 32,
        cat_embedding_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        unknown_dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 20,
        batch_size: int = 1024,
        patience: int = 3,
        random_state: int = 42,
    ) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.n_categories = n_categories
        self.item_categories = (
            item_categories
            if item_categories is not None
            else np.full(n_items + 1, n_categories, dtype="int64")
        )
        self.known_users = (
            known_users if known_users is not None else np.ones(n_users, dtype=bool)
        )
        self.known_items = (
            known_items if known_items is not None else np.ones(n_items, dtype=bool)
        )
        self.embedding_dim = embedding_dim
        self.cat_embedding_dim = cat_embedding_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.unknown_dropout = unknown_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.contract_version = CONTRACT_VERSION
        self._scaler: StandardScaler | None = None
        self._net: _NCFNet | None = None

    def _route_ids(
        self, ids: np.ndarray, known: np.ndarray, vocab_size: int
    ) -> np.ndarray:
        """Roteia ids fora do vocabulário/treino para o índice unknown (D4)."""
        idx = ids.astype("int64")
        routed = np.full(idx.shape, vocab_size, dtype="int64")
        valid = (idx >= 0) & (idx < vocab_size)
        routed[valid] = np.where(known[idx[valid]], idx[valid], vocab_size)
        return routed

    def _transform_cont(self, X: np.ndarray) -> np.ndarray:
        """log1p nas colunas de cauda pesada — aplicado antes do scaler."""
        cont = X[:, len(ID_COLS) :].astype("float64", copy=True)
        cont[:, _LOG_SCALE_IDX] = np.log1p(np.maximum(cont[:, _LOG_SCALE_IDX], 0.0))
        return cont

    def _split_matrix(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Separa a matriz do contrato em (users, items, cats, cont escalado)."""
        assert self._scaler is not None, "Chame fit() antes."
        users = self._route_ids(X[:, 0], self.known_users, self.n_users)
        items = self._route_ids(X[:, 1], self.known_items, self.n_items)
        cats = self.item_categories[items]
        cont = self._scaler.transform(self._transform_cont(X)).astype("float32")
        return users, items, cats, cont

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> NCFRecommender:
        """Treina a rede com early stopping por métrica de VALIDAÇÃO (FR-010).

        Args:
            X: Matriz de treino na ordem do contrato de features.
            y: Rótulos binários (1 = positivo, 0 = negativo amostrado).
            X_val: Matriz de validação (mesma ordem). Se fornecida, o early
                stopping usa o ROC-AUC de validação; senão, a loss de treino.
            y_val: Rótulos de validação.
        """
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self._scaler = StandardScaler().fit(self._transform_cont(X))
        users, items, cats, cont = self._split_matrix(X)
        users, items, cats = self._apply_unknown_dropout(users, items)
        dataset = RetailRocketDataset(users, items, cont, y, cats=cats)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        self._net = _NCFNet(
            self.n_users,
            self.n_items,
            self.n_categories,
            self.embedding_dim,
            self.cat_embedding_dim,
            len(CONT_COLS),
            self.hidden_dims,
            self.dropout,
        )
        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self._run_early_stopping(loader, optimizer, X_val, y_val)
        return self

    def _apply_unknown_dropout(
        self, users: np.ndarray, items: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Roteia uma fração dos ids de treino para o índice unknown (D4).

        Sem isso o embedding "unknown" nunca recebe gradiente e usuários/itens
        cold-start seriam pontuados com um vetor aleatório. Com o dropout de
        id, o unknown aprende o comportamento do "usuário/item médio".
        """
        if self.unknown_dropout <= 0:
            return users, items, self.item_categories[items]
        rng = np.random.default_rng(self.random_state)
        users = users.copy()
        items = items.copy()
        users[rng.random(len(users)) < self.unknown_dropout] = self.n_users
        items[rng.random(len(items)) < self.unknown_dropout] = self.n_items
        return users, items, self.item_categories[items]

    def _validation_metric(
        self,
        train_loss: float,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> float:
        """Métrica a maximizar no early stopping: val-AUC ou -loss de treino."""
        if X_val is None or y_val is None:
            return -train_loss
        return float(roc_auc_score(y_val, self.predict_proba(X_val)))

    def _run_early_stopping(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> None:
        """Loop de treino guardando o melhor estado pela métrica de validação."""
        assert self._net is not None
        loss_fn = nn.BCEWithLogitsLoss()
        best_metric, no_improve = -float("inf"), 0
        best_state = copy.deepcopy(self._net.state_dict())
        for _ in range(self.epochs):
            loss = self._train_epoch(loader, optimizer, loss_fn)
            metric = self._validation_metric(loss, X_val, y_val)
            if metric > best_metric + _MIN_IMPROVEMENT:
                best_metric, no_improve = metric, 0
                best_state = copy.deepcopy(self._net.state_dict())
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        self._net.load_state_dict(best_state)

    def _train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
    ) -> float:
        """Processa um epoch completo e retorna a loss média."""
        assert self._net is not None
        self._net.train()
        total = 0.0
        for users, items, cats, cont, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(self._net(users, items, cats, cont), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / max(len(loader), 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna a probabilidade de interação positiva por par (user, item)."""
        assert self._net is not None, "Chame fit() antes de predict_proba()"
        self._net.eval()
        X = np.asarray(X, dtype="float32")
        users, items, cats, cont = self._split_matrix(X)
        out = np.empty(len(X), dtype="float32")
        with torch.no_grad():
            for start in range(0, len(X), _PREDICT_BATCH):
                end = start + _PREDICT_BATCH
                logits = self._net(
                    torch.from_numpy(users[start:end]),
                    torch.from_numpy(items[start:end]),
                    torch.from_numpy(cats[start:end]),
                    torch.from_numpy(cont[start:end]),
                )
                out[start:end] = torch.sigmoid(logits).numpy()
        return out

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Retorna predições binárias (0 ou 1) com base no threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)
