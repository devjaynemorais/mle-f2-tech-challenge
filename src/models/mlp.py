"""Modelo principal: MLP treinado com PyTorch.

Registrado na Factory com o nome "mlp".
Usa BCEWithLogitsLoss para classificação binária e early stopping
para evitar overfitting.
"""

import numpy as np
import torch
import torch.nn as nn

from src.models.base import ModelFactory, RecommenderBase


class _MLPNet(nn.Module):
    """Rede feed-forward interna (não usada diretamente fora deste módulo).

    Estrutura: Linear → ReLU → Dropout → ... → Linear (1 saída por amostra)
    """

    def __init__(
        self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev, dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        layers.append(nn.Linear(prev, 1))  # saída: 1 logit por amostra
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@ModelFactory.register("mlp")
class MLPRecommender(RecommenderBase):
    """MLP para recomendação binária (interagiu ou não com o item).

    Args:
        input_dim: Número de features de entrada.
        hidden_dims: Tamanhos das camadas ocultas.
        dropout: Taxa de dropout aplicada após cada camada oculta.
        lr: Taxa de aprendizado.
        epochs: Número máximo de épocas de treino.
        batch_size: Tamanho do mini-batch.
        patience: Épocas sem melhora antes do early stopping.
        random_state: Semente para reprodutibilidade.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        patience: int = 5,
        random_state: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self._net: _MLPNet | None = None  # criado no fit()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRecommender":
        """Treina a rede com early stopping.

        Args:
            X: Matriz de features (n_amostras, input_dim).
            y: Rótulos binários (n_amostras,).
        """
        torch.manual_seed(self.random_state)

        # Converte para tensores PyTorch
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        y_t = torch.tensor(np.array(y), dtype=torch.float32)

        self._net = _MLPNet(self.input_dim, self.hidden_dims, self.dropout)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Early stopping: para quando a loss não melhora por `patience` épocas
        best_loss, no_improve = float("inf"), 0
        for _ in range(self.epochs):
            loss = self._train_epoch(loader, optimizer, loss_fn)
            if loss < best_loss:
                best_loss, no_improve = loss, 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        return self

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
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(self._net(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / len(loader)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna a probabilidade de interação (valor entre 0 e 1)."""
        assert self._net is not None, "Chame fit() antes de predict_proba()"
        self._net.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.array(X), dtype=torch.float32)
            return torch.sigmoid(self._net(X_t)).numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Retorna predições binárias (0 ou 1) com base no threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)
