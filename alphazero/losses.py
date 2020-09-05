import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from typing import Union, Dict, Any
from abc import ABC, abstractmethod


class Loss(nn.Module):
    @abstractmethod
    def forward(self):
        ...

    @abstractmethod
    def get_info(self):
        ...

    def __repr__(self):
        ...


class AlphaZeroLoss(Loss):
    def __init__(self, policy_coeff: float, value_coeff: float, reduction: str) -> None:
        super().__init__()

        self.name = type(self).__name__

        self.policy_coeff = policy_coeff
        self.value_coeff = value_coeff
        self.reduction = reduction

    def _calculate_policy_loss(
        self, pi_prior_logits: torch.Tensor, pi_mcts: torch.Tensor
    ) -> torch.Tensor:
        # first we have to convert the probabilities to labels
        pi_mcts = pi_mcts.argmax(dim=1)
        pi_loss = F.cross_entropy(pi_prior_logits, pi_mcts, reduction=self.reduction)
        return pi_loss

    def _calculate_value_loss(
        self, V_hat: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(V_hat, V, reduction=self.reduction)

    def forward(
        self,
        pi_prior_logits: torch.Tensor,
        pi_mcts: torch.Tensor,
        V_hat: torch.Tensor,
        V: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        policy_loss = self.policy_coeff * self._calculate_policy_loss(
            pi_prior_logits, pi_mcts
        )
        value_loss = self.value_coeff * self._calculate_value_loss(V_hat, V)
        loss = policy_loss + value_loss
        return {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss}

    def get_info(self) -> Dict[str, Any]:
        info = {"name": self.name}
        info.update(
            {
                key: getattr(self, key)
                for key in vars(self)
                if not key.startswith("_") and not key.startswith("training")
            }
        )
        return info

    def __repr__(self) -> str:
        return f"{self.name}: c_policy={self.policy_coeff}, c_value={self.value_coeff}, reduction={self.reduction}"


class A0CLoss(Loss):
    def __init__(
        self,
        tau: float,
        policy_coeff: float,
        alpha: float,
        value_coeff: float,
        reduction: str,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.policy_coeff = policy_coeff
        self.alpha = alpha
        self.value_coeff = value_coeff
        self.reduction = reduction

    def _calculate_policy_loss(
        self, log_probs: torch.Tensor, log_counts: torch.Tensor
    ) -> torch.Tensor:

        with torch.no_grad():
            # calculate scaling term
            log_diff = log_probs - self.tau * log_counts

        # multiple with log_probs gradient
        policy_loss = torch.einsum("ni, ni -> n", log_diff, log_probs)

        if self.reduction == "mean":
            return policy_loss.mean()
        else:
            return policy_loss.sum()

    def _calculate_value_loss(
        self, V_hat: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(V_hat, V, reduction=self.reduction)

    def _calculate_entropy_loss(self, entropy: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return -entropy.mean()
        else:
            return -entropy.sum()

    def forward(
        self,
        log_probs: torch.Tensor,
        log_counts: torch.tensor,
        entropy: torch.Tensor,
        V: torch.Tensor,
        V_hat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        policy_loss = self.policy_coeff * self._calculate_policy_loss(
            log_probs, log_counts
        )
        value_loss = self.value_coeff * self._calculate_value_loss(V_hat, V)
        entropy_loss = self.alpha * self._calculate_entropy_loss(entropy)
        loss = policy_loss + entropy_loss + value_loss
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
        }

    def get_info(self) -> Dict[str, Any]:
        info = {"name": self.name}
        info.update(
            {
                key: getattr(self, key)
                for key in vars(self)
                if not key.startswith("_") and not key.startswith("training")
            }
        )
        return info

    def __repr__(self) -> str:
        return f"{self.name}: c_policy={self.policy_coeff}, c_value={self.value_coeff}, reduction={self.reduction}"


class A0CLossTuned(A0CLoss):
    def __init__(
        self,
        action_dim: int,
        lr: float,
        tau: float,
        policy_coeff: float,
        value_coeff: float,
        reduction: str,
    ) -> None:
        self.tau = tau
        self.policy_coeff = policy_coeff
        self.value_coeff = value_coeff
        self.reduction = reduction

        # set target entropy to -|A|
        self.target_entropy = -action_dim
        # initialize alpha to 1
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().item()
        # for simplicity: Use the same optimizer settings as for the neural network
        self.a_optimizer = Adam([self.log_alpha], lr=lr)
        super().__init__(self.tau, self.policy_coeff, self.alpha, self.value_coeff, self.reduction)

    def _update_alpha(self, entropy: torch.Tensor) -> torch.Tensor:
        # we don't want to backprop through the network here
        a_entropy = entropy.detach()
        # calculate loss for entropy regularization parameter
        alpha_loss = (-self.log_alpha * (a_entropy + self.target_entropy)).mean()
        # optimize and set values
        self.a_optimizer.zero_grad()
        alpha_loss.backward()
        self.a_optimizer.step()
        self.alpha = self.log_alpha.exp().item()
        return alpha_loss.detach().cpu()

    def forward(
        self,
        log_probs: torch.Tensor,
        log_counts: torch.tensor,
        entropy: torch.Tensor,
        V: torch.Tensor,
        V_hat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        policy_loss = self.policy_coeff * self._calculate_policy_loss(
            log_probs, log_counts
        )
        value_loss = self.value_coeff * self._calculate_value_loss(V_hat, V)
        entropy_loss = self.alpha * self._calculate_entropy_loss(entropy)
        loss = policy_loss + entropy_loss + value_loss
        alpha_loss = self._update_alpha(entropy)
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            "alpha_loss": alpha_loss,
        }

    def get_info(self) -> Dict[str, Any]:
        info = {"name": self.name}
        info.update(
            {
                key: getattr(self, key)
                for key in vars(self)
                if not key.startswith("_") and not key.startswith("training")
            }
        )
        return info

    def __repr__(self) -> str:
        return f"{self.name}: c_policy={self.policy_coeff}, c_value={self.value_coeff}, reduction={self.reduction}"
