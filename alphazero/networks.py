from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class Network(ABC, nn.Module):

    @abstractmethod
    def predict_V(self):
        ...

    @abstractmethod
    def predict_pi(self):
        ...

class NetworkDiscrete(Network):
    def __init__(
        self, state_dim: int, action_dim: int, n_hidden_layers: int, n_hidden_units: int
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units

        layers = []
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(n_hidden_units, n_hidden_units))
            layers.append(nn.ELU())

        self.in_layer = nn.Linear(self.state_dim, n_hidden_units)

        self.hidden = nn.Sequential(*layers)

        self.policy_head = nn.Linear(n_hidden_units, self.action_dim)
        self.value_head = nn.Linear(n_hidden_units, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        # no need for softmax, can be computed directly from cross-entropy loss
        pi_hat = self.policy_head(x)
        V_hat = self.value_head(x)
        return pi_hat, V_hat

    @torch.no_grad()
    def predict_V(self, x: torch.Tensor) -> np.array:
        self.eval()
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        V_hat = self.value_head(x)
        self.train()
        return V_hat.detach().cpu().numpy()

    @torch.no_grad()
    def predict_pi(self, x: torch.Tensor) -> np.array:
        self.eval()
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        pi_hat = F.softmax(self.policy_head(x), dim=-1)
        self.train()
        return pi_hat.detach().cpu().numpy()


class NetworkContinuous(Network):
    def __init__(
        self, state_dim: int, action_dim: int, n_hidden_layers: int, n_hidden_units: int
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units

        layers = []
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(n_hidden_units, n_hidden_units))
            layers.append(nn.ELU())

        self.in_layer = nn.Linear(self.state_dim, n_hidden_units)

        self.hidden = nn.Sequential(*layers)

        self.policy_head = nn.Linear(n_hidden_units, self.action_dim)
        self.value_head = nn.Linear(n_hidden_units, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        # no need for softmax, can be computed directly from cross-entropy loss
        pi_hat = self.policy_head(x)
        V_hat = self.value_head(x)
        return pi_hat, V_hat

    @torch.no_grad()
    def predict_V(self, x: torch.Tensor) -> np.array:
        self.eval()
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        V_hat = self.value_head(x)
        self.train()
        return V_hat.detach().cpu().numpy()

    # TODO: Implement
    @torch.no_grad()
    def predict_pi(self, x: torch.Tensor) -> np.array:
        self.eval()
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        pi_hat = F.softmax(self.policy_head(x), dim=-1)
        self.train()
        return pi_hat.detach().cpu().numpy()