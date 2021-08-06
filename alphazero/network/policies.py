from typing import ClassVar, List, Optional, Tuple, Callable, Union, cast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from alphazero.network.distributions import SquashedNormal, GeneralizedBeta
from alphazero.network.utils import (
    _map_nonlinearities,
    _process_str,
)

__all__ = [
    "make_policy",
    "DiagonalNormalPolicy",
    "DiagonalGMMPolicy",
    "GeneralizedBetaPolicy",
    "DiscretePolicy",
]

class Policy(nn.Module):
    """Base policy class.

    The base policy is responsible for instanting the linear layers and value head.
    It also defines some interface functions.

    Parameters
    ----------
    representation_dim : int
        Dimensions of the input representation.
    action_dim : int
        Number of dimensions for the action space.
    distribution : str
        Distribution that is parameterized by the network.
        Allows the following options:
            - "normal": Normal distribution.
            - "tanhsquashed", "tanhsquashednormal": Normal distribution with samples squashed in (-1, 1).
            - "generalizedsquashed", "generalizedsquashednormal": Normal distribution with samples squashed in (-c, c).
            - "beta", "generalizedbeta": Beta distribution with transformed support on (-c, c).
    action_bound : Optional[float]
        Bounds for the action space. Can be either float or None.
    hidden_dimensions : List[int]
        Specify the number of hidden neurons for each respective hidden layer of the network. Cannot be empty.
    nonlinearity : str
        Nonlinearity used between hidden layers. Options are:
            - "relu": https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU  .
            - "leakyrelu": https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU.
            - "relu6": https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html#torch.nn.ReLU6.
            - "silu": https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU.
            - "elu": https://pytorch.org/docs/stable/generated/torch.nn.ELU.html#torch.nn.ELU.
            - "hardswish": https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html#torch.nn.Hardswish.
    layernorm : bool
        If True, the network is regularized with layer normalization after each liner layer.
        This may increase performance, see https://arxiv.org/pdf/1709.06560.pdf for info.
    log_param_min : int
        Lower bound for learned log parameters.
    log_param_max : int
        Upper bound for learned log parameters.
    """

    # member type annotations
    state_dim: int
    action_dim: int
    action_bound: Optional[float]
    log_param_min: float
    log_param_max: float
    hidden_layers: int
    hidden_dimensions: List[int]
    trunk: nn.Sequential
    value_head: nn.Linear

    def __init__(
        self,
        representation_dim: int,
        action_dim: int,
        action_bound: Optional[float],
        hidden_dimensions: List[int],
        nonlinearity: str,
        layernorm: bool,
        log_param_min: float,
        log_param_max: float,
    ):
        super().__init__()
        self.state_dim = representation_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        # boundaries for the log standard deviation to increae training stability
        self.log_param_min = log_param_min
        self.log_param_max = log_param_max

        assert hidden_dimensions, "Hidden dimensions can't be empty."
        self.hidden_dimensions = hidden_dimensions
        self.hidden_layers = len(hidden_dimensions)

        activation: Callable[..., nn.Module] = _map_nonlinearities(nonlinearity)
        self.layernorm = layernorm

        # generate neural network except distribution heads
        layers = [
            nn.Linear(self.state_dim, hidden_dimensions[0]),
            activation(inplace=True),
        ]
        if layernorm:
            layers.append(nn.LayerNorm(normalized_shape=hidden_dimensions[0]))

        if 1 < self.hidden_layers:
            for i, hidden_dim in enumerate(hidden_dimensions[:-1]):
                hid = [
                    nn.Linear(hidden_dim, hidden_dimensions[i + 1]),
                    activation(inplace=True),
                ]
                if layernorm:
                    hid.append(nn.LayerNorm(normalized_shape=hidden_dimensions[i + 1]))
                layers.extend(hid)

        self.trunk = nn.Sequential(*layers)

        self.value_head = nn.Linear(hidden_dimensions[-1], 1)

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            String representation of this instance.
        """
        components: int = getattr(self, "num_components", 1)
        return (
            f"class={type(self).__name__}, distribution={self.distribution_type}, components={components}, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, action_bounds={self.bounds}, "
            f"log_std_bounds={self.log_param_bounds}, hidden_layers={self.hidden_layers}, hidden_units={self.hidden_dimensions}, "
            f"nonlinearity={type(self.trunk[1]).__name__}, layernorm={self.layernorm}"
        )

    @property
    def bounds(self) -> np.ndarray:
        if self.action_bound is None:
            return np.array([-np.inf, np.inf], dtype=np.float32)
        else:
            return np.array([-self.action_bound, self.action_bound], dtype=np.float32)

    @torch.no_grad()
    def get_train_data(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def sample_action(self, x: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    @torch.no_grad()
    def predict_V(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        x = self.trunk(x)
        V_hat = self.value_head(x)
        self.train()
        return V_hat.detach().cpu().numpy()


class DiscretePolicy(nn.Module):
    """Base policy class.

    The base policy is responsible for instanting the linear layers and value head.
    It also defines some interface functions.

    Parameters
    ----------
    representation_dim : int
        Dimensions of the input representation.
    action_dim : int
        Number of dimensions for the action space.
    distribution : str
        Distribution that is parameterized by the network.
        Allows the following options:
            - "normal": Normal distribution.
            - "tanhsquashed", "tanhsquashednormal": Normal distribution with samples squashed in (-1, 1).
            - "generalizedsquashed", "generalizedsquashednormal": Normal distribution with samples squashed in (-c, c).
            - "beta", "generalizedbeta": Beta distribution with transformed support on (-c, c).
    action_bound : Optional[float]
        Bounds for the action space. Can be either float or None.
    hidden_dimensions : List[int]
        Specify the number of hidden neurons for each respective hidden layer of the network. Cannot be empty.
    nonlinearity : str
        Nonlinearity used between hidden layers. Options are:
            - "relu": https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU  .
            - "leakyrelu": https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU.
            - "relu6": https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html#torch.nn.ReLU6.
            - "silu": https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU.
            - "elu": https://pytorch.org/docs/stable/generated/torch.nn.ELU.html#torch.nn.ELU.
            - "hardswish": https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html#torch.nn.Hardswish.
    layernorm : bool
        If True, the network is regularized with layer normalization after each liner layer.
        This may increase performance, see https://arxiv.org/pdf/1709.06560.pdf for info.
    log_param_min : int
        Lower bound for learned log parameters.
    log_param_max : int
        Upper bound for learned log parameters.
    """

    # member type annotations
    state_dim: int
    action_dim: int
    num_actions: int
    hidden_layers: int
    hidden_dimensions: List[int]
    trunk: nn.Sequential
    value_head: nn.Linear

    # class variable
    distribution_type: ClassVar[str] = "Categorical"

    def __init__(
        self,
        representation_dim: int,
        action_dim: int,
        num_actions: int,
        hidden_dimensions: List[int],
        nonlinearity: str,
        layernorm: bool,
    ):
        super().__init__()
        self.state_dim = representation_dim
        self.action_dim = action_dim
        self.num_actions = num_actions

        assert hidden_dimensions, "Hidden dimensions can't be empty."
        self.hidden_dimensions = hidden_dimensions
        self.hidden_layers = len(hidden_dimensions)
        self.distribution = D.Categorical

        activation: Callable[..., nn.Module] = _map_nonlinearities(nonlinearity)
        self.layernorm = layernorm

        # generate neural network except distribution heads
        layers = [
            nn.Linear(self.state_dim, hidden_dimensions[0]),
            activation(inplace=True),
        ]
        if layernorm:
            layers.append(nn.LayerNorm(normalized_shape=hidden_dimensions[0]))

        if 1 < self.hidden_layers:
            for i, hidden_dim in enumerate(hidden_dimensions[:-1]):
                hid = [
                    nn.Linear(hidden_dim, hidden_dimensions[i + 1]),
                    activation(inplace=True),
                ]
                if layernorm:
                    hid.append(nn.LayerNorm(normalized_shape=hidden_dimensions[i + 1]))
                layers.extend(hid)

        self.trunk = nn.Sequential(*layers)

        self.value_head = nn.Linear(hidden_dimensions[-1], 1)

        self.dist_head = nn.Linear(hidden_dimensions[-1], num_actions)

    def __repr__(self) -> str:
        """
        Returns
        -------
        str
            String representation of this instance.
        """
        return (
            f"class={type(self).__name__}, distribution={self.distribution_type}, num_actions={self.num_actions}, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"hidden_layers={self.hidden_layers}, hidden_units={self.hidden_dimensions}, "
            f"nonlinearity={type(self.trunk[1]).__name__}, layernorm={self.layernorm}"
        )

    def _get_dist_params(
        self, x: torch.Tensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Returns the learned paremters of the distribution.

        Parameters
        ----------
        x : torch.FloatTensor
            Input state tensor.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
            Distribution mean (mu), Distribution standard deviation (sigma), State value estimate (V_hat).
        """
        x = self.trunk(x)
        V_hat = self.value_head(x)

        # dist_head returns a tensor of shape [batch_size, 2*action_dim]
        # split this tensor along the last dimension into parameters for mu and sigma
        pi_logits = self.dist_head(x)

        return pi_logits, V_hat

    def forward(self, x: torch.FloatTensor) -> Tuple[D.Categorical, torch.FloatTensor]:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.FloatTensor
            Input state tensor.

        Returns
        -------
        Tuple[Normallike, torch.FloatTensor]
            Normal or squashed Normal distribution (dist), State value estimate (V_hat).
        """
        pi_logits, V_hat = self._get_dist_params(x)

        dist = D.Categorical(logits=pi_logits)

        # samples from dist have shape [batch_size, action_dim]
        return dist, V_hat

    def get_train_data(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        pi_logits, V_hat = self._get_dist_params(states)

        # This creates an independent distribution for each action possibility
        # so that the batch_shape of the distribution is identical to the shape of actions
        # It's needed so that the log_probs are of the proper shape [batch_size, num_actions]
        # else this throws since the distribution's batch_shape=[batch_shape] doesn't match
        # the shape of the actions tensor, which is [batch_size, num_actions]
        num_actions = actions.shape[1]
        pi_hat = D.Categorical(
            logits=pi_logits.unsqueeze(dim=1).repeat((1, num_actions, 1))
        )
        log_probs = pi_hat.log_prob(actions)

        entropy = pi_hat.entropy()

        return log_probs, entropy, V_hat

    @torch.no_grad()
    def predict_V(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        _, V_hat = self(x)
        self.train()
        return V_hat.detach().cpu().numpy()

    @torch.no_grad()
    def predict_pi(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        logits, _ = self._get_dist_params(x)
        self.train()
        return F.softmax(logits, dim=-1).detach().cpu().numpy()


class DiagonalNormalPolicy(Policy):
    """Policy class for factorized normal distributions.

    Learns parameters for a factorized normal distribution of types
    Normal, TanhSquashedNormal or GeneralizedSquashedNormal.
    Factorized means that a conditionally independent (given a state) 1D Normal distribution is
    learned for each dimension of the action space instead of a Multivariate Normal.

    Parameters
    ----------
    representation_dim : int
        Dimensions of the input representation.
    action_dim : int
        Number of dimensions for the action space.
    distribution : str
        Distribution that is parameterized by the network. Has to be a Normallike  distribution.
        Allows the following options:
            - "normal": Normal distribution.
            - "tanhsquashed", "tanhsquashednormal": Normal distribution with samples squashed in (-1, 1).
            - "generalizedsquashed", "generalizedsquashednormal": Normal distribution with samples squashed in (-c, c).
    action_bound : Optional[float]
        Bounds for the action space. Can be either float or None.
    hidden_dimensions : List[int]
        Specify the number of hidden neurons for each respective hidden layer of the network. Cannot be empty.
    nonlinearity : str
        Nonlinearity used between hidden layers. Options are:
            - "relu": https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU  .
            - "leakyrelu": https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU.
            - "relu6": https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html#torch.nn.ReLU6.
            - "silu": https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU.
            - "elu": https://pytorch.org/docs/stable/generated/torch.nn.ELU.html#torch.nn.ELU.
            - "hardswish": https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html#torch.nn.Hardswish.
    layernorm : bool
        If True, the network is regularized with layer normalization after each liner layer.
        This may increase performance, see https://arxiv.org/pdf/1709.06560.pdf for info.
    log_param_min : int
        Lower bound for learned log standard deviation.
    log_param_max : int
        Upper bound for learned log standard deviation.
    """

    # member annotations
    state_dim: int
    action_dim: int
    action_bound: Optional[float]
    log_param_min: float
    log_param_max: float
    hidden_layers: int
    hidden_dimensions: List[int]
    trunk: nn.Sequential
    dist_head: nn.Linear
    value_head: nn.Linear

    # class variable
    policy_type: ClassVar[str] = "DiagonalNormal"

    def __init__(
        self,
        representation_dim: int,
        action_dim: int,
        action_bound: Optional[float],
        hidden_dimensions: List[int],
        nonlinearity: str,
        layernorm: bool,
        log_param_min: float,
        log_param_max: float,
    ):

        super().__init__(
            representation_dim=representation_dim,
            action_dim=action_dim,
            action_bound=action_bound,
            hidden_dimensions=hidden_dimensions,
            nonlinearity=nonlinearity,
            layernorm=layernorm,
            log_param_min=log_param_min,
            log_param_max=log_param_max,
        )

        self.dist_head = nn.Linear(hidden_dimensions[-1], 2 * self.action_dim)

    def forward(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Returns the learned paremters of the distribution.

        Parameters
        ----------
        x : torch.FloatTensor
            Input state tensor.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
            Distribution mean (mu), Distribution standard deviation (sigma), State value estimate (V_hat).
        """
        x = self.trunk(x)
        V_hat = self.value_head(x)

        # dist_head returns a tensor of shape [batch_size, 2*action_dim]
        # split this tensor along the last dimension into parameters for mu and sigma
        mu, log_std = self.dist_head(x).chunk(2, dim=-1)

        # Learning the log_std_dev is a trick for numerical stability
        # Since the stddev > 0, we can learn the log and then exponentiate
        # constrain log_std inside [log_param_min, log_param_max]
        log_std = torch.clamp(log_std, min=self.log_param_min, max=self.log_param_max)
        sigma = log_std.exp()

        return mu, sigma, V_hat

    def get_train_data(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma, V_hat = self(states)
        # This aligns the distribution batch_shape with the number of actions at the root
        # It can be thought of as generating num_actions identical normal distributions for each agent
        # and then sampling the log_prob for action from the distribution
        # num_actions = actions.shape[-1]
        # mu = mu.expand((-1, num_actions))
        # sigma = sigma.expand((-1, num_actions))

        normal: Union[D.Normal, SquashedNormal]
        if self.action_bound:
            normal = SquashedNormal(mu, sigma, self.action_bound)
        else:
            normal = D.Normal(mu, sigma)

        log_probs = normal.log_prob(actions)
        entropy = -log_probs.mean(dim=-1)

        return log_probs, entropy, V_hat

    @torch.no_grad()
    def sample_action(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        mu, sigma, _ = self(x)
        normal: Union[D.Normal, SquashedNormal]
        if self.action_bound:
            normal = SquashedNormal(mu, sigma, self.action_bound)
        else:
            normal = D.Normal(mu, sigma)
        action = normal.sample()
        self.train()
        return action.detach().cpu().numpy()


class DiagonalGMMPolicy(Policy):
    """Policy class for learning a factorized GMM.

    Learns a 1D GMM for each dimension of the action space.
    The components of the GMM are either Normal or squashed Normal.

    Parameters
    ----------
    representation_dim : int
        Dimensions of the input representation.
    action_dim : int
        Number of dimensions for the action space.
    distribution : str
        Distribution that is parameterized by the network. Has to be Normallike.
        Allows the following options:
            - "normal": Normal distribution.
            - "tanhsquashed", "tanhsquashednormal": Normal distribution with samples squashed in (-1, 1).
            - "generalizedsquashed", "generalizedsquashednormal": Normal distribution with samples squashed in (-c, c).
    num_components : int
        Number of mixture components.
    action_bound : Optional[float]
        Bounds for the action space. Can be either float or None.
    hidden_dimensions : List[int]
        Specify the number of hidden neurons for each respective hidden layer of the network. Cannot be empty.
    nonlinearity : str
        Nonlinearity used between hidden layers. Options are:
            - "relu": https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU  .
            - "leakyrelu": https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU.
            - "relu6": https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html#torch.nn.ReLU6.
            - "silu": https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU.
            - "elu": https://pytorch.org/docs/stable/generated/torch.nn.ELU.html#torch.nn.ELU.
            - "hardswish": https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html#torch.nn.Hardswish.
    layernorm : bool
        If True, the network is regularized with layer normalization after each liner layer.
        This may increase performance, see https://arxiv.org/pdf/1709.06560.pdf for info.
    log_param_min : int
        Lower bound for learned log standard deviations.
    log_param_max : int
        Upper bound for learned log standard deviations.
    """

    # member annotations
    state_dim: int
    action_dim: int
    action_bound: Optional[float]
    log_param_min: float
    log_param_max: float
    hidden_layers: int
    hidden_dimensions: List[int]
    num_components: int
    trunk: nn.Sequential
    dist_head: nn.Linear
    value_head: nn.Linear

    # class variable
    policy_type: ClassVar[str] = "DiagonalGMM"

    def __init__(
        self,
        representation_dim: int,
        action_dim: int,
        action_bound: Optional[float],
        num_components: int,
        hidden_dimensions: List[int],
        nonlinearity: str,
        layernorm: bool,
        log_param_min: float,
        log_param_max: float,
    ):

        super().__init__(
            representation_dim=representation_dim,
            action_dim=action_dim,
            action_bound=action_bound,
            hidden_dimensions=hidden_dimensions,
            nonlinearity=nonlinearity,
            layernorm=layernorm,
            log_param_min=log_param_min,
            log_param_max=log_param_max,
        )

        self.num_components = num_components

        # calculate the number of parameters needed for the GMM
        # 2 comes from each distribution being specifiec by 2 parameters
        dist_params = num_components * (2 * self.action_dim + 1)
        self.dist_head = nn.Linear(hidden_dimensions[-1], dist_params)

    def forward(
        self, x: torch.FloatTensor
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Returns the learned paremters of the distribution.

        Parameters
        ----------
        x : torch.FloatTensor
            Input state tensor.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
            Distribution mean (mu), Distribution standard deviation (sigma),
            Logits for the categorical distribution parameterizing the components (log_coeffs),
            State value estimate (V_hat).
        """
        x = self.trunk(x)
        V_hat = self.value_head(x)

        # mixture_params is a tensor of shape [batch_size, num_agents, 2*action_dim*num_components + num_components]
        # the elements in the first term (2*action_dim*num_components) are the parameters for the mixture components
        # the elements in the second term (+ num_components) are the mixture coefficients
        mixture_params = self.dist_head(x)
        # get mixture parameters and reorder to [batch_size, num_agents, 2*num_components, action_dim]
        dist_params = mixture_params[
            ..., : self.num_components * 2 * self.action_dim
        ].view(x.shape[0], -1)
        # get the num_components last tensor elements as logits for the mixture coefficients
        log_coeff = mixture_params[..., -self.num_components :]
        # split the dist_params along the middle  dimension (2*num_components) into means and log stddevs
        mu, log_std = dist_params.chunk(2, dim=-1)

        # Learning the log_std_dev is a trick for numerical stability
        # Since the stddev > 0, we can learn the log and then exponentiate
        # constrain log_std inside [log_param_min, log_param_max]
        log_std = torch.clamp(log_std, min=self.log_param_min, max=self.log_param_max)
        sigma = log_std.exp()

        return mu, sigma, log_coeff, V_hat

    def get_train_data(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma, log_coeff, V_hat = self(states)

        # We need num_actions identical gmms to sample log_probs for each action
        num_actions = actions.shape[-1]
        mu = mu.unsqueeze(dim=1).expand((-1, num_actions, -1))
        sigma = sigma.unsqueeze(dim=1).expand((-1, num_actions, -1))
        log_coeff = log_coeff.unsqueeze(dim=1).expand((-1, num_actions, -1))
        mix = D.Categorical(logits=log_coeff)

        component: Union[D.Normal, SquashedNormal]
        if self.action_bound:
            component = SquashedNormal(mu, sigma, self.action_bound)
        else:
            component = D.Normal(mu, sigma)
        gmm = D.MixtureSameFamily(mix, component)
        log_probs = gmm.log_prob(actions)
        entropy = -log_probs.mean(dim=-1)

        return log_probs, entropy, V_hat

    @torch.no_grad()
    def sample_action(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        mu, sigma, log_coeff, _ = self(x)
        mix = D.Categorical(logits=log_coeff)
        component: Union[D.Normal, SquashedNormal]
        if self.action_bound:
            component = SquashedNormal(mu, sigma, self.action_bound)
        else:
            component = D.Normal(mu, sigma)
        gmm = D.MixtureSameFamily(mix, component)
        action = gmm.sample()
        self.train()
        return action.detach().cpu().numpy()


class GeneralizedBetaPolicy(Policy):
    """Policy class for a generalized Beta distribution.

    The beta distribution used by this class is generalized in that it has support
    [-c, c] instead of [0,1].
    This is achieved via a location-scale transformation (2c)x - c, where c are the desired bounds.
    Since both parameters alpha, beta > 0, the log-learning-trick for the Normal standard deviation
    is applied to both parameters.

    Parameters
    ----------
    representation_dim : int
        Dimensions of the input representation.
    action_dim : int
        Number of dimensions for the action space.
    action_bound : Optional[float]
        Bounds for the action space. Can be either float or None.
    hidden_dimensions : List[int]
        Specify the number of hidden neurons for each respective hidden layer of the network. Cannot be empty.
    nonlinearity : str
        Nonlinearity used between hidden layers. Options are:
            - "relu": https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU  .
            - "leakyrelu": https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU.
            - "relu6": https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html#torch.nn.ReLU6.
            - "silu": https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU.
            - "elu": https://pytorch.org/docs/stable/generated/torch.nn.ELU.html#torch.nn.ELU.
            - "hardswish": https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html#torch.nn.Hardswish.
    layernorm : bool
        If True, the network is regularized with layer normalization after each liner layer.
        This may increase performance, see https://arxiv.org/pdf/1709.06560.pdf for info.
    log_param_min : int
        Lower bound for learned log_alpha and log_beta.
    log_param_max : int
        Upper bound for learned log_alpha and log_beta.
    """

    # member annotations
    state_dim: int
    action_dim: int
    action_bound: float
    log_param_min: float
    log_param_max: float
    hidden_layers: int
    hidden_dimensions: List[int]
    trunk: nn.Sequential
    dist_head: nn.Linear
    value_head: nn.Linear

    # class variable
    policy_type: ClassVar[str] = "GeneralizedBeta"

    def __init__(
        self,
        representation_dim: int,
        action_dim: int,
        action_bound: float,
        hidden_dimensions: List[int],
        nonlinearity: str,
        layernorm: bool,
        log_param_min: float,
        log_param_max: float,
    ):

        assert action_bound, "Beta policy needs action bounds specified."

        super().__init__(
            representation_dim=representation_dim,
            action_dim=action_dim,
            action_bound=action_bound,
            hidden_dimensions=hidden_dimensions,
            nonlinearity=nonlinearity,
            layernorm=layernorm,
            log_param_min=log_param_min,
            log_param_max=log_param_max,
        )

        self.dist_head = nn.Linear(hidden_dimensions[-1], 2 * self.action_dim)

    def forward(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Returns the learned paremters of the distribution.

        Parameters
        ----------
        x : torch.FloatTensor
            Input state tensor.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
            Alpha parameter (alpha), Beta parameter (beta), State value estimate (V_hat).
        """
        x = self.trunk(x)
        V_hat = self.value_head(x)

        # create distribution parameters
        dist_params = self.dist_head(x)

        # Use the log_std_dev trick for alpha and beta
        # since both alpha > 0 and beta > 0
        dist_params = torch.clamp(
            dist_params, min=self.log_param_min, max=self.log_param_max
        )
        alpha, beta = dist_params.exp().chunk(2, dim=-1)

        return alpha, beta, V_hat

    def get_train_data(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha, beta, V_hat = self(states)

        # ensure that the distribution batch_shape fits the number of actions taken for
        # each agent at the root
        num_actions = actions.shape[-1]
        alpha = alpha.expand(-1, num_actions)
        beta = beta.expand(-1, num_actions)
        beta_dist = GeneralizedBeta(alpha, beta, self.action_bound)
        log_probs = beta_dist.log_prob(actions)
        entropy = -log_probs.mean(dim=-1)

        return log_probs, entropy, V_hat

    @torch.no_grad()
    def sample_action(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        alpha, beta, _ = self(x)
        beta_dist = D.Beta(alpha, beta)
        action = beta_dist.sample()
        self.train()
        return action.detach().cpu().numpy()


def make_policy(
    representation_dim: int,
    action_dim: int,
    distribution: str,
    hidden_dimensions: List[int],
    nonlinearity: str,
    num_components: Optional[int] = None,
    num_actions: Optional[int] = None,
    action_bound: Optional[float] = None,
    layernorm: bool = False,
    log_param_min: float = -5,
    log_param_max: float = 2,
) -> Union[
    DiscretePolicy, DiagonalNormalPolicy, DiagonalGMMPolicy, GeneralizedBetaPolicy
]:
    """Constructs a policy network from a given config.

    The following config keys need to be specified:
        - "representation_dim": int
        - "action_dim": int
        - "distribution": str
        - "num_components": int
        - "action_bound": float
        - "hidden_dimensions": List[int]
        - "nonlinearity": str
        - "layernorm": bool
        - "log_param_min": Optional[float]
        - "log_param_max": Optional[float]

    Parameters
    ----------
    config : Dict[str, Any]
        Policy config dictionary.

    Returns
    -------
    Union[DiagonalNormalPolicy, DiagonalGMMPolicy, GeneralizedBetaPolicy]
        Policy network intance.
    """

    # basic config string preprocessing to ensure mapping works later
    distribution = _process_str(distribution)
    nonlinearity = _process_str(nonlinearity)

    if distribution == "discrete":
        return DiscretePolicy(
            representation_dim=representation_dim,
            action_dim=action_dim,
            num_actions=cast(int, num_actions),
            hidden_dimensions=hidden_dimensions,
            nonlinearity=nonlinearity,
            layernorm=layernorm,
        )
    elif distribution == "beta":
        return GeneralizedBetaPolicy(
            representation_dim=representation_dim,
            action_dim=action_dim,
            action_bound=cast(float, action_bound),
            hidden_dimensions=hidden_dimensions,
            nonlinearity=nonlinearity,
            layernorm=layernorm,
            log_param_min=log_param_min,
            log_param_max=log_param_max,
        )
    else:
        if 1 < num_components:
            return DiagonalGMMPolicy(
                representation_dim=representation_dim,
                action_dim=action_dim,
                num_components=num_components,
                action_bound=action_bound,
                hidden_dimensions=hidden_dimensions,
                nonlinearity=nonlinearity,
                layernorm=layernorm,
                log_param_min=log_param_min,
                log_param_max=log_param_max,
            )
        else:
            return DiagonalNormalPolicy(
                representation_dim=representation_dim,
                action_dim=action_dim,
                action_bound=action_bound,
                hidden_dimensions=hidden_dimensions,
                nonlinearity=nonlinearity,
                layernorm=layernorm,
                log_param_min=log_param_min,
                log_param_max=log_param_max,
            )
