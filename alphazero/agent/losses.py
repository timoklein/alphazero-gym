import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam

from typing import Dict, Union
from abc import abstractmethod


class Loss(nn.Module):
    @abstractmethod
    def forward(self):
        ...

    @abstractmethod
    def _calculate_policy_loss(self) -> torch.Tensor:
        ...

    @abstractmethod
    def _calculate_value_loss(
        self, V_hat: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        ...


class AlphaZeroLoss(Loss):

    policy_coeff: float
    value_coeff: float
    reduction: str

    def __init__(self, policy_coeff: float, value_coeff: float, reduction: str) -> None:
        super().__init__()

        self.name = type(self).__name__

        self.policy_coeff = policy_coeff
        self.value_coeff = value_coeff
        self.reduction = reduction

    def _calculate_policy_loss(  # type: ignore[override]
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

    def forward(  # type: ignore[override]
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


# FIXME: Parameterize loss clamping
class A0CLoss(Loss):

    tau: float
    policy_coeff: float
    alpha: Union[float, torch.Tensor]
    value_coeff: float
    reduction: str

    def __init__(
        self,
        tau: float,
        policy_coeff: float,
        alpha: Union[float, torch.Tensor],
        value_coeff: float,
        reduction: str,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.policy_coeff = policy_coeff
        self.alpha = alpha
        self.value_coeff = value_coeff
        self.reduction = reduction

    def _calculate_policy_loss(  # type: ignore[override]
        self, log_probs: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:

        with torch.no_grad():
            # calculate scaling term
            log_diff = log_probs - self.tau * torch.log(counts)

        # multiply with log_probs gradient
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
            return entropy.mean()
        else:
            return entropy.sum()

    def forward(  # type: ignore[override]
        self,
        log_probs: torch.Tensor,
        counts: torch.Tensor,
        entropy: torch.Tensor,
        V: torch.Tensor,
        V_hat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        policy_loss = self.policy_coeff * self._calculate_policy_loss(log_probs, counts)
        value_loss = self.value_coeff * self._calculate_value_loss(V_hat, V)
        entropy_loss = self.alpha * self._calculate_entropy_loss(entropy)
        loss = policy_loss + entropy_loss + value_loss
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
        }


class A0CLossTuned(A0CLoss):

    tau: float
    policy_coeff: float
    alpha: torch.Tensor
    value_coeff: float
    reduction: str
    clip: float
    device: torch.device
    log_alpha: torch.Tensor
    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        action_dim: int,
        alpha_init: float,
        lr: float,
        tau: float,
        policy_coeff: float,
        value_coeff: float,
        reduction: str,
        grad_clip: float,
        device: str,
    ) -> None:
        self.clip = grad_clip
        self.device = torch.device(device)

        # set target entropy to -|A|
        self.target_entropy = -action_dim
        # initialize alpha to 1
        self.log_alpha = torch.tensor(
            np.log(alpha_init),
            requires_grad=True,
            device=self.device,
            dtype=torch.float32,
        )

        self.alpha = self.log_alpha.exp()

        self.optimizer = Adam([self.log_alpha], lr=lr)

        # for simplicity: Use the same optimizer settings as for the neural network
        super().__init__(
            tau=tau,
            policy_coeff=policy_coeff,
            alpha=self.alpha,
            value_coeff=value_coeff,
            reduction=reduction,
        )

    def _update_alpha(self, entropy: torch.Tensor) -> torch.Tensor:
        self.log_alpha.grad = None
        # calculate loss for entropy regularization parameter
        alpha_loss = (self.alpha * (entropy - self.target_entropy).detach()).mean()
        alpha_loss.backward()

        if self.clip:
            clip_grad_norm(self.log_alpha, self.clip)
        self.optimizer.step()

        self.alpha = self.log_alpha.exp()

        return alpha_loss.detach().cpu()

    def forward(  # type: ignore[override]
        self,
        log_probs: torch.Tensor,
        counts: torch.Tensor,
        entropy: torch.Tensor,
        V: torch.Tensor,
        V_hat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        policy_loss = self.policy_coeff * self._calculate_policy_loss(log_probs, counts)
        value_loss = self.value_coeff * self._calculate_value_loss(V_hat, V)
        entropy_loss = self.alpha.detach().item() * self._calculate_entropy_loss(
            entropy
        )
        loss = policy_loss + entropy_loss + value_loss
        alpha_loss = self._update_alpha(entropy)
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            "alpha_loss": alpha_loss,
        }
