import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
from torch.optim import Adam

from typing import Dict, Union
from abc import abstractmethod


class Loss(nn.Module):
    """ABC for the loss classes."""

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
    """AlphaZero loss.

    This class implements the loss function from the AlphaZero paper.
    It ONLY works in discrete settings.

    Attributes
    ----------
    policy_coeff: float
        Scaling factor for the policy component.
    value_coeff: float
        Scaling factor for the value component.
    reduction: str
        Defines how the loss is reduced to a scalar. Can be either "sum" or "mean".
    """

    policy_coeff: float
    value_coeff: float
    reduction: str

    def __init__(self, policy_coeff: float, value_coeff: float, reduction: str) -> None:
        """Constructor.

        Parameters
        ----------
        policy_coeff: float
            Scaling factor for the policy component.
        value_coeff: float
            Scaling factor for the value component.
        reduction: str
            Defines how the loss is reduced to a scalar. Can be either "sum" or "mean".
        """
        super().__init__()

        self.name = type(self).__name__

        self.policy_coeff = policy_coeff
        self.value_coeff = value_coeff
        self.reduction = reduction

    def _calculate_policy_loss(  # type: ignore[override]
        self, pi_prior_logits: torch.Tensor, pi_mcts: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the policy loss.

        The policy loss is the cross-entropy between the network probability distribution
        and the action with the highest visitation counts.

        Parameters
        ----------
        pi_prior_logits: torch.Tensor
            Prior distribution over the available actions from the network.
        pi_mcts: torch.Tensor
            Improved MCTS policy for the same state.

        Returns
        -------
        torch.Tensor
            Policy loss as scalar tensor.
        """
        # first we have to convert the probabilities to labels
        pi_mcts = pi_mcts.argmax(dim=1)
        pi_loss = F.cross_entropy(pi_prior_logits, pi_mcts, reduction=self.reduction)
        return pi_loss

    def _calculate_value_loss(
        self, V_hat: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the value loss of AlphaZero.

        The value loss is the mean squared error between the value estimate of the
        neural network and an improved value target produced by the MCTS.

        Parameters
        ----------
        V_hat: torch.Tensor
            Value estimates from the neural network for the training state.
        V: torch.Tensor
            V
            Improved value targets for that state.

        Returns
        -------
        torch.Tensor
            Value loss as scalar tensor.
        """
        return F.mse_loss(V_hat, V, reduction=self.reduction)

    def forward(  # type: ignore[override]
        self,
        pi_prior_logits: torch.Tensor,
        pi_mcts: torch.Tensor,
        V_hat: torch.Tensor,
        V: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Short description

        Longer description

        Parameters
        ----------
        pi_prior_logits: torch.Tensor
            Prior probabilities for all actions from the neural network.
        pi_mcts: torch.Tensor
            Normalized MCTS visitation counts for the selected actions.
        V_hat: torch.Tensor
            Neural network value estimates.
        V: torch.Tensor
            Value targets.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary holding the scalar loss values as values and the name of the
            component as key.
        """
        policy_loss = self.policy_coeff * self._calculate_policy_loss(
            pi_prior_logits, pi_mcts
        )
        value_loss = self.value_coeff * self._calculate_value_loss(V_hat, V)
        loss = policy_loss + value_loss
        return {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss}


class A0CLoss(Loss):
    """Implementation of the A0C loss.

    A0C is an extension of AlphaZero for continuous action spaces. It formulates a continuous
    training target and adds an entropy loss component to prevent the distribution from
    collapsing. More information is in the paper: https://arxiv.org/pdf/1805.09613.pdf.

    Attributes
    ----------
    tau: float
        Temperature parameter for the log-visitation counts in the policy loss.
    policy_coeff: float
        Scaling factor for the policy component of the loss.
    alpha: Union[float, torch.Tensor]
        Scaling factor for the entropy regularization term.
    value_coeff: float
        Scaling factor for the value component of the loss.
    reduction: str
        Defines how the loss is reduced to a scalar. Can be either "sum" or "mean".
    """

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
        """Constructor.

        Parameters
        ----------
        tau: float
            Temperature parameter for the log-visitation counts in the policy loss.
        policy_coeff: float
            Scaling factor for the policy component of the loss.
        alpha: Union[float, torch.Tensor]
            Scaling factor for the entropy regularization term.
        value_coeff: float
            Scaling factor for the value component of the loss.
        reduction: str
            Defines how the loss is reduced to a scalar. Can be either "sum" or "mean".
        """
        super().__init__()
        self.tau = tau
        self.policy_coeff = policy_coeff
        self.alpha = alpha
        self.value_coeff = value_coeff
        self.reduction = reduction

    def _calculate_policy_loss(  # type: ignore[override]
        self, log_probs: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        """Implements the A0C policy loss.

        The A0C policy loss uses the REINFORCE trick to move the continuous network
        distribution closer to a distribution specified by the normalized visitation counts.
        More information is in the paper: https://arxiv.org/pdf/1805.09613.pdf.

        Parameters
        ----------
        log_probs: torch.Tensor
            Action log-probabilities from the network's policy distribution.
        counts: torch.Tensor
            Action visitation counts

        Returns
        -------
        torch.Tensor
            Reduced policy loss.
        """
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
        """Calculate the value loss of A0C.

        The value loss is the same as in the original AlphaZero paper.

        Parameters
        ----------
        V_hat: torch.Tensor
            Value estimates from the neural network for the training state.
        V: torch.Tensor
            V
            Improved value targets for that state.

        Returns
        -------
        torch.Tensor
            Value loss as scalar tensor.
        """
        return F.mse_loss(V_hat, V, reduction=self.reduction)

    def _calculate_entropy_loss(self, entropy: torch.Tensor) -> torch.Tensor:
        """Calculate the entropy regularization term.

        The entropy of a distribution can be approximated through the action log-probabilities.
        Note: Analytical computation is not possible for the squashed normal distribution
        or a GMM policy.

        Parameters
        ----------
        entropy: torch.Tensor
            Entropy as output from the policy network.

        Returns
        -------
        torch.Tensor
            Entropy regularization term as scalar Tensor.
        """
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
        """Calculate the A0C loss.

        Parameters
        ----------
        log_probs: torch.Tensor
            Action log probabilities from the network policy given a state.
        counts: torch.Tensor
            Action visitation counts.
        entropy: torch.Tensor
            Approximate Entropy of the neural network distribution for a given state.
        V_hat: torch.Tensor
            Neural network value estimates.
        V: torch.Tensor
            Value targets.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary holding the loss component name as keys and the loss value for the component
            as scalar Tensor.
        """
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
    """Tuned version of the A0C loss using automatic entropy tuning from SAC.

    This class is the same as the A0C loss except that the temperature for the entropy
    regularization term is adjusted automatically over the course of the training.
    Since the temperature parameter can never be negative, log-alpha is learned and
    then exponentiated.
    More information is in the second SAC paper: https://arxiv.org/pdf/1812.05905.pdf.

    Attributes
    ----------
    tau: float
        Temperature parameter for the log-visitation counts in the policy loss.
    policy_coeff: float
        Scaling factor for the policy component of the loss.
    alpha: Union[float, torch.Tensor]
        Scaling factor for the entropy regularization term.
    value_coeff: float
        Scaling factor for the value component of the loss.
    reduction: str
        Defines how the loss is reduced to a scalar. Can be either "sum" or "mean".
    clip: float
        Gradient clipping value for the alpha loss.
    device: torch.device
        Device for the loss. Can be either "cpu" or "cuda".
    log_alpha: torch.Tensor
        Log parameter that is actually learned.
    optimizer: torch.optim.Optimizer
        Torch optimizer for adjusting the log-alpha parameter.
    """

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
        """Constructor.

        Parameters
        ----------
        action_dim: int
            Dimensionality of the action space. Used as target for tuning alpha.
        alpha_init: float
            Initial value for alpha.
        lr: float
            Alpha optimizer learning rate.
        tau: float
            Temperature parameter for the log-visitation counts in the policy loss.
        policy_coeff: float
            Scaling factor for the policy component of the loss.
        value_coeff: float
            Scaling factor for the value component of the loss.
        reduction: str
            Defines how the loss is reduced to a scalar. Can be either "sum" or "mean".
        grad_clip: float
            Gradient clipping value for the alpha loss.
        device: torch.device
            Device for the loss. Can be either "cpu" or "cuda".
        """
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
        """Perform an update state for the entropy regularization term temperature parameter
        alpha.

        Parameters
        ----------
        entropy: torch.Tensor
            Approximate policy distribution entropy from the network.

        Returns
        -------
        torch.Tensor
            Alpha loss as scalar Tensor.
        """
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
        """Calculate the A0C loss and perform an alpha update step step.

        Parameters
        ----------
        log_probs: torch.Tensor
            Action log probabilities from the network policy given a state.
        counts: torch.Tensor
            Action visitation counts.
        entropy: torch.Tensor
            Approximate Entropy of the neural network distribution for a given state.
        V_hat: torch.Tensor
            Neural network value estimates.
        V: torch.Tensor
            Value targets.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary holding the loss component name as keys and the loss value for the component
            as scalar Tensor.
        """
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
