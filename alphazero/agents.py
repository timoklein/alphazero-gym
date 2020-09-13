from alphazero.losses import A0CLoss
import torch
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm
import numpy as np
import gym
import hydra
from omegaconf.dictconfig import DictConfig


from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from abc import ABC, abstractmethod

from torch.optim import optimizer

from .networks import NetworkContinuous, NetworkDiscrete
from .helpers import stable_normalizer
from .buffers import ReplayBuffer


class Agent(ABC):
    def __init__(
        self,
        network_cfg: DictConfig,
        loss_cfg: DictConfig,
        mcts_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        grad_clip: float,
        device: str,
    ) -> None:

        self.device = torch.device(device)
        self.nn = hydra.utils.instantiate(network_cfg).to(self.device)
        self.mcts = hydra.utils.instantiate(mcts_cfg, model=self.nn)
        self.loss = hydra.utils.instantiate(loss_cfg).to(self.device)
        self.optimizer = hydra.utils.instantiate(
            optimizer_cfg, params=self.nn.parameters()
        )

        self.clip = grad_clip

    @abstractmethod
    def act(self):
        ...

    @abstractmethod
    def update(self):
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

    def reset_mcts(self, root_state: np.array) -> None:
        self.mcts.root_node = None
        self.mcts.root_state = root_state

    def train(self, buffer: ReplayBuffer) -> float:
        buffer.reshuffle()
        running_loss = defaultdict(float)
        for epoch in range(1):
            for batches, obs in enumerate(buffer):
                loss = self.update(obs)
                for key in loss.keys():
                    running_loss[key] += loss[key]
        for val in running_loss.values():
            val /= batches + 1
        return running_loss


class DiscreteAgent(Agent):
    def __init__(
        self,
        is_atari: bool,
        network_cfg: DictConfig,
        mcts_cfg: DictConfig,
        loss_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        grad_clip: float,
        temperature: float,
        device: str,
    ) -> None:

        super().__init__(
            network_cfg=network_cfg,
            loss_cfg=loss_cfg,
            mcts_cfg=mcts_cfg,
            optimizer_cfg=optimizer_cfg,
            grad_clip=grad_clip,
            device=device,
        )
        self.is_atari = is_atari

        # initialize values
        self.temperature = temperature

    def act(
        self,
        Env: gym.Env,
        mcts_env: gym.Env,
        simulation: bool = False,
        deterministic: bool = False,
    ) -> Tuple[int, np.array, np.array, np.array]:

        self.mcts.search(Env=Env, mcts_env=mcts_env, simulation=simulation)
        state, actions, counts, V = self.mcts.return_results()

        # Get MCTS policy
        pi = stable_normalizer(counts, self.temperature)

        # sample an action from the policy or pick best action if deterministic
        action = pi.argmax() if deterministic else np.random.choice(len(pi), p=pi)

        return action, state, actions, counts, V

    def mcts_forward(self, action: int, node: np.array) -> None:
        self.mcts.forward(action, node)

    def update(self, obs: Tuple[np.array, np.array, np.array]) -> Dict[str, float]:

        # zero out gradients
        for param in self.nn.parameters():
            param.grad = None

        states, actions, counts, V_target = obs
        states_tensor = torch.from_numpy(states).float().to(self.device)
        values_tensor = (
            torch.from_numpy(V_target).unsqueeze(dim=1).float().to(self.device)
        )

        if isinstance(self.loss, A0CLoss):
            actions_tensor = torch.from_numpy(actions).float().to(self.device)
            # regularize the counts to always be greater than 0
            # this prevents the logarithm from producing nans in the next step
            counts += 1
            log_counts_tensor = torch.log(torch.from_numpy(counts).float()).to(
                self.device
            )

            log_probs, entropy, V_hat = self.nn.get_train_data(
                states_tensor, actions_tensor
            )
            loss_dict = self.loss(
                log_probs=log_probs,
                log_counts=log_counts_tensor,
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
        network_cfg: DictConfig,
        mcts_cfg: DictConfig,
        loss_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        grad_clip: float,
        device: str,
    ) -> None:

        super().__init__(
            network_cfg=network_cfg,
            loss_cfg=loss_cfg,
            mcts_cfg=mcts_cfg,
            optimizer_cfg=optimizer_cfg,
            grad_clip=grad_clip,
            device=device,
        )

    @property
    def action_limit(self) -> float:
        return self.nn.act_limit

    def act(
        self, Env: gym.Env, mcts_env: gym.Env, simulation: bool = False,
    ) -> Tuple[int, np.array, np.array, np.array]:

        self.mcts.search(Env=Env, mcts_env=mcts_env, simulation=simulation)
        state, actions, counts, V_hat = self.mcts.return_results()

        # select the action that was visited most
        action = actions[counts.argmax()][np.newaxis]

        return action, state, actions, counts, V_hat

    def update(
        self, obs: Tuple[np.array, np.array, np.array, np.array]
    ) -> Dict[str, float]:

        # zero out gradients
        for param in self.nn.parameters():
            param.grad = None

        states, actions, counts, V_target = obs

        actions_tensor = torch.from_numpy(actions).float().to(self.device)
        states_tensor = torch.from_numpy(states).float().to(self.device)
        values_tensor = (
            torch.from_numpy(V_target).unsqueeze(dim=1).float().to(self.device)
        )

        log_counts_tensor = torch.log(torch.from_numpy(counts).float()).to(self.device)

        log_probs, entropy, V_hat = self.nn.get_train_data(
            states_tensor, actions_tensor
        )

        loss_dict = self.loss(
            log_probs=log_probs,
            log_counts=log_counts_tensor,
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

