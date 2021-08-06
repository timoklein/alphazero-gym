import random
from collections import defaultdict
from typing import Any, Dict, Tuple, Union
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import numpy as np
import gym
import hydra
from omegaconf.dictconfig import DictConfig

from alphazero.helpers import stable_normalizer
from alphazero.agent.buffers import ReplayBuffer
from alphazero.agent.losses import A0CLoss, A0CLossTuned, AlphaZeroLoss
from alphazero.search.mcts import MCTSContinuous, MCTSDiscrete
from alphazero.network.policies import (
    DiscretePolicy,
    DiagonalNormalPolicy,
    DiagonalGMMPolicy,
    GeneralizedBetaPolicy,
)


class Agent(ABC):
    """Abstract base class for the AlphaZero agent.

    Defines the interface and some common methods for the discrete and continuous agent.

    Attributes
    ----------
    device: torch.device
        Torch device. Can be either CPU or cuda.
    nn: Union[DiscretePolicy, DiagonalNormalPolicy, DiagonalGMMPolicy, GeneralizedBetaPolicy]
        Neural network policy used by this agent.
    mcts: Union[MCTSDiscrete, MCTSContinuous]
        Tree search algorithm. Continuous MCTS used progressive widening.
    loss: Union[AlphaZeroLoss, A0CLoss, A0CLossTuned]
        Loss object to train the policy.
    optimizer: torch.optim.Optimizer
        Pytorch optimizer object for performing gradient descent.
    final_selection: str
        String indicating how the final action should be chosen. Can be either "max_visit"
        or "max_value".
    train_epochs: int
        Number of training epochs per episode.
    clip: float
        Value for gradient clipping.
    """

    def __init__(
        self,
        policy_cfg: DictConfig,
        loss_cfg: DictConfig,
        mcts_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        final_selection: str,
        train_epochs: int,
        grad_clip: float,
        device: str,
    ) -> None:
        """Initializer for common attributes of all agent instances.

        Parameters
        ----------
        policy_cfg: DictConfig
            Hydra configuration object for the policy.
        loss_cfg: DictConfig
            Hydra configuration object for the loss.
        mcts_cfg: DictConfig
            Hydra configuration object for the MCTS.
        optimizer_cfg: DictConfig
            Hydra configuration object for the SGD optimizer.
        final_selection: str
            String identifier for the final selection policy. Can be either "max_visit"
            or "max_value".
        train_epochs: int
            Number of training epochs per episode step.
        grad_clip: float
            Gradient clipping value.
        device: str
            Device used to train the network. Can be either "cpu" or "cuda".
        """

        # instantiate network
        self.device = torch.device(device)
        self.nn = hydra.utils.call(policy_cfg).to(torch.device(device))
        self.mcts = hydra.utils.instantiate(mcts_cfg, model=self.nn)
        self.loss = hydra.utils.instantiate(loss_cfg).to(self.device)
        self.optimizer = hydra.utils.instantiate(
            optimizer_cfg, params=self.nn.parameters()
        )

        self.final_selection = final_selection
        self.train_epochs = train_epochs
        self.clip = grad_clip

    @abstractmethod
    def act(
        self,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interface for the act method (interaction with the environment)."""
        ...

    @abstractmethod
    def update(
        self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Interface for a single gradient descent update step."""
        ...

    @property
    def action_dim(self) -> int:
        """Returns the dimensionality of the action space as int."""
        return self.nn.action_dim

    @property
    def state_dim(self) -> int:
        """Returns the dimensionality of the state space as int."""
        return self.nn.state_dim

    @property
    def n_hidden_layers(self) -> int:
        """Returns the number of hidden layers in the policy network as int."""
        return self.nn.n_hidden_layers

    @property
    def n_hidden_units(self) -> int:
        """Computes the total number of hidden units and returns them as int."""
        return self.nn.n_hidden_units

    @property
    def n_rollouts(self) -> int:
        """Returns the number of MCTS search iterations per environment step."""
        return self.mcts.n_rollouts

    @property
    def learning_rate(self) -> float:
        """Float learning rate of the optimizer."""
        return self.optimizer.lr

    @property
    def c_uct(self) -> float:
        """Constant (float) in the MCTS selection policy weighing the exploration term (UCTS constant)."""
        return self.mcts.c_uct

    @property
    def gamma(self) -> float:
        """Returns the MCTS discount factor as float."""
        return self.mcts.gamma

    def reset_mcts(self, root_state: np.ndarray) -> None:
        """Reset the MCTS by setting the root node to a target environment state.

        Parameters
        ----------
        root_state: np.ndarray
            Environment state defining the new root node.
        """
        self.mcts.root_node = None
        self.mcts.root_state = root_state

    def train(self, buffer: ReplayBuffer) -> Dict[str, Any]:
        """Implementation of a training loop for the neural network.

        The training loop is executed after each environment episode. It is the same
        for both continuous and discrete agents. Differences are in the update method
        which must be implemented for each agent individually.

        Parameters
        ----------
        buffer: ReplayBuffer
            Instance of the replay buffer class containing the training experiences.

        Returns
        -------
        Dict[str, Any]
            Dictionary holding the values of all loss components as float. Keys are the names
            of the loss components.
        """
        buffer.reshuffle()
        running_loss: Dict[str, Any] = defaultdict(float)
        for epoch in range(self.train_epochs):
            for batches, obs in enumerate(buffer):
                loss = self.update(obs)
                for key in loss.keys():
                    running_loss[key] += loss[key]
        for val in running_loss.values():
            val = val / (batches + 1)
        return running_loss


class DiscreteAgent(Agent):
    """Implementation of an AlphaZero agent for discrete action spaces.

    The Discrete agent handles execution of the MCTS as well as network training.
    It interacts with the environment through the act method which executes the search
    and returns the training data.
    Implements an update step for the discrete algorithm is in the update method.

    Attributes
    ----------
    temperature : float
        Temperature parameter for the normalization procedure in the action selection.
    """

    def __init__(
        self,
        policy_cfg: DictConfig,
        mcts_cfg: DictConfig,
        loss_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        final_selection: str,
        train_epochs: int,
        grad_clip: float,
        temperature: float,
        device: str,
    ) -> None:
        """Constructor for the discrete agent.

        Delegates the initialization of components to the ABC constructor.

        Parameters
        ----------
        policy_cfg: DictConfig
            Hydra configuration object for the policy.
        loss_cfg: DictConfig
            Hydra configuration object for the loss.
        mcts_cfg: DictConfig
            Hydra configuration object for the MCTS.
        optimizer_cfg: DictConfig
            Hydra configuration object for the SGD optimizer.
        final_selection: str
            String identifier for the final selection policy. Can be either "max_visit"
            or "max_value".
        train_epochs: int
            Number of training epochs per episode step.
        grad_clip: float
            Gradient clipping value.
        temperature: float
            Temperature parameter for normalizing the visit counts in the final
            selection policy.
        device: str
            Device used to train the network. Can be either "cpu" or "cuda".
        """

        super().__init__(
            policy_cfg=policy_cfg,
            loss_cfg=loss_cfg,
            mcts_cfg=mcts_cfg,
            optimizer_cfg=optimizer_cfg,
            final_selection=final_selection,
            train_epochs=train_epochs,
            grad_clip=grad_clip,
            device=device,
        )

        assert isinstance(self.mcts, MCTSDiscrete)

        # initialize values
        self.temperature = temperature

    def act(  # type: ignore[override]
        self,
        Env: gym.Env,
        deterministic: bool = False,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main interface method for the agent to interact with the environment.

        The act method wraps execution of the MCTS search and final action selection.
        It also returns the statistics at the root node for network training.
        The choice of the action to be executed can be either based on visitation counts
        or on action values. Through the deterministic flag it can be specified if this
        choice is samples from the visitation/action value distribution.

        Parameters
        ----------
        Env: gym.Env
            Gym environment from which the MCTS should be executed.
        deterministic: bool = False
            If True, the action with the highest visitation count/action value is executed
            in the environment. If false, the final action is samples from the visitation count
            or action value distribution.

        Returns
        -------
        Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the action to be executed in the environment and root node
            training information. Elements are:
                - action: MCTS-improved action to be executed in the environment.
                - state: Root node state vector.
                - actions: Root node child actions.
                - counts: Visitation counts at the root node.
                - Qs: Action values at the root node.
                - V: Value target returned from the MCTS.
        """
        self.mcts.search(Env=Env)
        state, actions, counts, Qs, V = self.mcts.return_results(self.final_selection)

        if self.final_selection == "max_value":
            # select final action based on max Q value
            pi = stable_normalizer(Qs, self.temperature)
            action = pi.argmax() if deterministic else np.random.choice(len(pi), p=pi)
        else:
            # select the final action based on visit counts
            pi = stable_normalizer(counts, self.temperature)
            action = pi.argmax() if deterministic else np.random.choice(len(pi), p=pi)

        return action, state, actions, counts, Qs, V

    def mcts_forward(self, action: int, node: np.ndarray) -> None:
        """Moves the MCTS root node to the actually selected node.

        Using the selected node as future root node implements tree reuse.

        Parameters
        ----------
        action: int
            Action that has been selected in the environment.
        node: np.ndarray
            Environment state for the new root node.
        """
        self.mcts.forward(action, node)

    def update(
        self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Performs a gradient descent update step.

        This is the main training method for the neural network. Given a batch of observations
        from the replay buffer, it uses the network, optimizer and loss attributes of
        this instance to perform a single update step within the train method.

        Parameters
        ----------
        obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Batch of observations. Contains:
                - states: Root node states.
                - actions: Selected actions at each root node state.
                - counts: Visitation counts for the actions at each root state.
                - Qs: Action values at the root node (currently unused).
                - V_target: Improved MCTS value targets.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are the name of a loss component (full loss, policy loss, value loss, entropy loss)
            and the values are the scalar loss values.
        """

        # zero out gradients
        for param in self.nn.parameters():
            param.grad = None

        # Qs are currently unused in update setp
        states: np.ndarray
        actions: np.ndarray
        counts: np.ndarray
        V_target: np.ndarray
        states, actions, counts, _, V_target = obs
        states_tensor = torch.from_numpy(states).float().to(self.device)
        values_tensor = (
            torch.from_numpy(V_target).unsqueeze(dim=1).float().to(self.device)
        )

        if isinstance(self.loss, A0CLoss):
            actions_tensor = torch.from_numpy(actions).float().to(self.device)
            # regularize the counts to always be greater than 0
            # this prevents the logarithm from producing nans in the next step
            counts += 1
            counts_tensor = torch.from_numpy(counts).float().to(self.device)

            log_probs, entropy, V_hat = self.nn.get_train_data(
                states_tensor, actions_tensor
            )
            loss_dict = self.loss(
                log_probs=log_probs,
                counts=counts_tensor,
                entropy=entropy,
                V=values_tensor,
                V_hat=V_hat,
            )
        else:
            action_probs_tensor = F.softmax(
                torch.from_numpy(counts).float(), dim=-1
            ).to(self.device)
            pi_logits, V_hat = self.nn(states_tensor)
            loss_dict = self.loss(pi_logits, action_probs_tensor, V_hat, values_tensor)

        loss_dict["loss"].backward()

        if self.clip:
            clip_grad_norm(self.nn.parameters(), self.clip)

        self.optimizer.step()

        info_dict = {key: float(value) for key, value in loss_dict.items()}
        return info_dict


class ContinuousAgent(Agent):
    """Implementation of an A0C agent for continuous action spaces.

    The Continuous agent handles execution of the MCTS as well as network training.
    It interacts with the environment through the act method which executes the search
    and returns the training data.
    Implements an update step for the A0C loss in the update method.
    The differences between the continuous agent and the discrete agent are:
        - The continuous agent uses an MCTS with progressive widening.
        - Only the A0C loss and the tuned A0C loss work for this agent.
        - The policy network must use either a normal distribution, a GMM or a Beta distribution.

    Attributes
    ----------
    temperature : float
        Temperature parameter for the normalization procedure in the action selection.
    """

    def __init__(
        self,
        policy_cfg: DictConfig,
        mcts_cfg: DictConfig,
        loss_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        final_selection: str,
        epsilon: float,
        train_epochs: int,
        grad_clip: float,
        device: str,
    ) -> None:
        """Constructor for the discrete agent.

        Delegates the initialization of components to the ABC constructor.

        Parameters
        ----------
        policy_cfg: DictConfig
            Hydra configuration object for the policy.
        loss_cfg: DictConfig
            Hydra configuration object for the loss.
        mcts_cfg: DictConfig
            Hydra configuration object for the MCTS.
        optimizer_cfg: DictConfig
            Hydra configuration object for the SGD optimizer.
        final_selection: str
            String identifier for the final selection policy. Can be either "max_visit"
            or "max_value".
        epsilon: float
            Epsilon value for epsilon-greedy action selection. Epsilon-greedy is disabled
            when this value is set to 0.
        train_epochs: int
            Number of training epochs per episode step.
        grad_clip: float
            Gradient clipping value.
        device: str
            Device used to train the network. Can be either "cpu" or "cuda".
        """

        super().__init__(
            policy_cfg=policy_cfg,
            loss_cfg=loss_cfg,
            mcts_cfg=mcts_cfg,
            optimizer_cfg=optimizer_cfg,
            final_selection=final_selection,
            train_epochs=train_epochs,
            grad_clip=grad_clip,
            device=device,
        )

        self.epsilon = epsilon

    @property
    def action_limit(self) -> float:
        """Returns the action bound for this agent as float."""
        return self.nn.act_limit

    def epsilon_greedy(self, actions: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Epsilon-greedy implementation for the final action selection.

        Parameters
        ----------
        actions: np.ndarray
            Actions to choose from.
        values: np.ndarray
            Values according which the best action is selected. Can be either visitation
            counts or action values.

        Returns
        -------
        np.ndarray
            Action chosen according to epsilon-greedy.
        """
        if random.random() < self.epsilon:
            return np.random.choice(actions)[np.newaxis]
        else:
            return actions[values.argmax()][np.newaxis]

    def act(  # type: ignore[override]
        self,
        Env: gym.Env,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main interface method for the agent to interact with the environment.

        The act method wraps execution of the MCTS search and final action selection.
        It also returns the statistics at the root node for network training.
        The choice of the action to be executed can be either the most visited action or
        the action with the highest action value. If the epsilon > 0 is specified when
        instantiating this agent, actions are selected using the epsilon-greedy algorithm.

        Parameters
        ----------
        Env: gym.Env
            Gym environment from which the MCTS should be executed.

        Returns
        -------
        Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the action to be executed in the environment and root node
            training information. Elements are:
                - action: MCTS-improved action to be executed in the environment.
                - state: Root node state vector.
                - actions: Root node child actions.
                - counts: Visitation counts at the root node.
                - Qs: Action values at the root node.
                - V: Value target returned from the MCTS.
        """
        self.mcts.search(Env=Env)
        state, actions, counts, Qs, V = self.mcts.return_results(self.final_selection)

        if self.final_selection == "max_value":
            if self.epsilon == 0:
                # select the action with the best action value
                action = actions[Qs.argmax()][np.newaxis]
            else:
                action = self.epsilon_greedy(actions=actions, values=Qs)
        else:
            if self.epsilon == 0:
                # select the action that was visited most
                action = actions[counts.argmax()][np.newaxis]
            else:
                action = self.epsilon_greedy(actions=actions, values=counts)

        return action, state, actions, counts, Qs, V

    def update(
        self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Performs a gradient descent update step.

        This is the main training method for the neural network. Given a batch of observations
        from the replay buffer, it uses the network, optimizer and loss attributes of
        this instance to perform a single update step within the train method.

        Parameters
        ----------
        obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Batch of observations. Contains:
                - states: Root node states.
                - actions: Selected actions at each root node state.
                - counts: Visitation counts for the actions at each root state.
                - Qs: Action values at the root node (currently unused).
                - V_target: Improved MCTS value targets.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are the name of a loss component (full loss, policy loss, value loss, entropy loss)
            and the values are the scalar loss values.
        """

        # zero out gradients
        for param in self.nn.parameters():
            param.grad = None

        # Qs are currently unused in update
        states: np.ndarray
        actions: np.ndarray
        counts: np.ndarray
        V_target: np.ndarray
        states, actions, counts, _, V_target = obs

        actions_tensor = torch.from_numpy(actions).float().to(self.device)
        states_tensor = torch.from_numpy(states).float().to(self.device)
        counts_tensor = torch.from_numpy(counts).float().to(self.device)
        values_tensor = (
            torch.from_numpy(V_target).unsqueeze(dim=1).float().to(self.device)
        )

        log_probs, entropy, V_hat = self.nn.get_train_data(
            states_tensor, actions_tensor
        )

        loss_dict = self.loss(
            log_probs=log_probs,
            counts=counts_tensor,
            entropy=entropy,
            V=values_tensor,
            V_hat=V_hat,
        )

        loss_dict["loss"].backward()

        if self.clip:
            clip_grad_norm(self.nn.parameters(), self.clip)

        self.optimizer.step()

        info_dict = {key: float(value) for key, value in loss_dict.items()}
        return info_dict
