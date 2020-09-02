from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from abc import ABC, abstractmethod


class NetworkDiscrete(nn.Module):
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


class NetworkContinuous(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_space_low: float,
        action_space_high: float,
        n_hidden_layers: int,
        n_hidden_units: int,
        log_max: int = 2,
        log_min: int = -5,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.action_scale = torch.FloatTensor(
            (action_space_high - action_space_low) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_space_high + action_space_low) / 2.)

        self.LOG_STD_MAX = log_max
        self.LOG_STD_MIN = log_min

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units

        layers = []
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(n_hidden_units, n_hidden_units))
            layers.append(nn.ELU())

        self.in_layer = nn.Linear(self.state_dim, n_hidden_units)

        self.hidden = nn.Sequential(*layers)

        self.mean_head = nn.Linear(n_hidden_units, self.action_dim)
        self.std_head = nn.Linear(n_hidden_units, self.action_dim)
        self.value_head = nn.Linear(n_hidden_units, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        mean = self.mean_head(x)
        log_std = self.std_head(x)
        log_std = torch.tanh(log_std)
        # Trick from OpenAI spinning up to scale log standard deviation
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        V_hat = self.value_head(x)
        return mean, log_std, V_hat

    @torch.no_grad()
    def predict_V(self, x: torch.Tensor) -> np.array:
        self.eval()
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        V_hat = self.value_head(x)
        self.train()
        return V_hat.detach().cpu().numpy()

    def sample(self, x: torch.Tensor) -> Tuple[np.array, np.array, torch.Tensor, torch.Tensor]:
        mean, log_std, V_hat = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        # no need reparameterization trick (mean + std * N(0,1))
        x_t = normal.sample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        # This is the correction for tanh squashing of the log_std and actions
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +  1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action.detach().cpu().numpy(), mean.detach().cpu().numpy(), log_prob, V_hat
