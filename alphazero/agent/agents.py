import random
from collections import defaultdict
from typing import Any, Dict, Tuple
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
from alphazero.network.policies import make_policy
from alphazero.agent.losses import A0CLoss


class Agent(ABC):
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
        ...

    @abstractmethod
    def update(
        self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        ...

    @property
    def action_dim(self) -> int:
        return self.nn.action_dim

    @property
    def state_dim(self) -> int:
        return self.nn.state_dim

    @property
    def n_hidden_layers(self) -> int:
        return self.nn.n_hidden_layers

    @property
    def n_hidden_units(self) -> int:
        return self.nn.n_hidden_units

    @property
    def n_rollouts(self) -> int:
        return self.mcts.n_rollouts

    @property
    def learning_rate(self) -> float:
        return self.optimizer.lr

    @property
    def c_uct(self) -> float:
        return self.mcts.c_uct

    @property
    def gamma(self) -> float:
        return self.mcts.gamma

    def reset_mcts(self, root_state: np.ndarray) -> None:
        self.mcts.root_node = None
        self.mcts.root_state = root_state

    def train(self, buffer: ReplayBuffer) -> Dict[str, Any]:
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

        # initialize values
        self.temperature = temperature

    def act(  # type: ignore[override]
        self,
        Env: gym.Env,
        mcts_env: gym.Env,
        simulation: bool = False,
        deterministic: bool = False,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        self.mcts.search(Env=Env, mcts_env=mcts_env, simulation=simulation)
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
        self.mcts.forward(action, node)

    def update(
        self, obs: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:

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
        return self.nn.act_limit

    def epsilon_greedy(self, actions: np.ndarray, values: np.ndarray) -> np.ndarray:
        if random.random() < self.epsilon:
            return np.random.choice(actions)[np.newaxis]
        else:
            return actions[values.argmax()][np.newaxis]

    def act(  # type: ignore[override]
        self,
        Env: gym.Env,
        mcts_env: gym.Env,
        simulation: bool = False,
    ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        self.mcts.search(Env=Env, mcts_env=mcts_env, simulation=simulation)
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
