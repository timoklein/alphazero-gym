from numpy.lib.ufunclike import isposinf
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import gym
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from abc import ABC, abstractmethod

from torch.optim.rmsprop import RMSprop

from .networks import NetworkContinuous, NetworkDiscrete
from .mcts import MCTSContinuous, MCTSDiscrete
from .helpers import is_atari_game, check_space, stable_normalizer
from .buffers import ReplayBuffer


class Agent(ABC):
    @abstractmethod
    def act(self):
        ...

    @abstractmethod
    def calculate_loss(self):
        ...

    @abstractmethod
    def update(self):
        ...

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def load_checkpoint(self):
        ...

    @abstractmethod
    def save_checkpoint(self):
        ...


class AlphaZeroAgent(Agent):
    def __init__(
        self,
        Env: gym.Env,
        n_hidden_layers: int,
        n_hidden_units: int,
        value_loss_ratio: float,
        n_traces: int,
        lr: float,
        temperature: float,
        c_uct: float,
        gamma: float,
    ) -> None:
        # get info about the environment
        state_dim, self.state_discrete = check_space(Env.observation_space)
        self.state_dim = state_dim[0]
        self.action_dim, self.action_discrete = check_space(Env.action_space)

        # initialize values
        self.n_traces = n_traces
        self.c_uct = c_uct
        self.gamma = gamma
        self.lr = lr
        self.temperature = temperature
        self.value_loss_ratio = value_loss_ratio

        self.is_atari = is_atari_game(Env)

        self.nn = NetworkDiscrete(
            self.state_dim,
            self.action_dim,
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
        )
        self.optimizer = optim.RMSprop(
            self.nn.parameters(), lr=self.lr, alpha=0.9, eps=1e-07
        )

    @property
    def n_hidden_layers(self) -> int:
        return self.nn.n_hidden_layers

    @property
    def n_hidden_units(self) -> int:
        return self.nn.n_hidden_units

    def reset_mcts(self, root_state: np.array) -> None:
        self.mcts = MCTSDiscrete(
            model=self.nn,
            num_actions=self.nn.action_dim,
            is_atari=self.is_atari,
            gamma=self.gamma,
            c_uct=self.c_uct,
            root_state=root_state,
        )

    def act(
        self, Env: gym.Env, mcts_env: gym.Env, deterministic: bool = False
    ) -> Tuple[int, np.array, np.array, np.array]:
        self.mcts.search(n_traces=self.n_traces, Env=Env, mcts_env=mcts_env)
        state, pi, V = self.mcts.return_results(self.temperature)
        # sample an action from the policy or pick best action if deterministic
        action = pi.argmax() if deterministic else np.random.choice(len(pi), p=pi)
        return action, state, pi, V

    def mcts_forward(self, action: int, node: np.array) -> None:
        self.mcts.forward(action, node)

    def calculate_loss(
        self,
        pi_logits: torch.Tensor,
        V_hat: torch.tensor,
        V: torch.Tensor,
        pi: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # calculate policy loss from model logits
        # first we have to convert the probabilities to labels
        pi = pi.argmax(dim=1)
        pi_loss = F.cross_entropy(pi_logits, pi)
        # value loss
        v_loss = self.value_loss_ratio * F.mse_loss(V_hat, V)
        loss = pi_loss + v_loss
        return {"loss": loss, "policy_loss": pi_loss, "value_loss": v_loss}

    def update(self, obs: Tuple[np.array, np.array, np.array]) -> Dict[str, float]:
        self.optimizer.zero_grad()

        state_batch, V_batch, pi_batch = obs
        states_tensor = torch.from_numpy(state_batch).float()
        values_tensor = torch.from_numpy(V_batch).float()
        action_probs_tensor = torch.from_numpy(pi_batch).float()

        pi_logits, V_hat = self.nn(states_tensor)
        loss_dict = self.calculate_loss(
            pi_logits, V_hat, values_tensor, action_probs_tensor
        )
        loss_dict["loss"].backward()
        self.optimizer.step()

        loss_dict["loss"] = loss_dict["loss"].detach().item()
        loss_dict["policy_loss"] = loss_dict["policy_loss"].detach().item()
        loss_dict["value_loss"] = loss_dict["value_loss"].detach().item()
        return loss_dict

    def train(self, buffer: ReplayBuffer) -> float:
        buffer.reshuffle()
        running_loss = {"loss": 0, "policy_loss": 0, "value_loss": 0}
        for epoch in range(1):
            for batches, obs in enumerate(buffer):
                loss = self.update(obs)
                running_loss["loss"] += loss["loss"]
                running_loss["policy_loss"] += loss["policy_loss"]
                running_loss["value_loss"] += loss["value_loss"]
        for val in running_loss.values():
            val /= batches + 1
        return running_loss

    # TODO: Need to map location to device
    # self.policy.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    def load_checkpoint(self, name: str) -> None:
        model_path = (Path("models") / name).with_suffix(".tar")
        checkpoint = torch.load(model_path)
        self.nn = NetworkDiscrete(
            state_dim=checkpoint["env_state_dim"],
            action_dim=checkpoint["env_action_dim"],
            n_hidden_layers=checkpoint["n_hidden_layers"],
            n_hidden_units=checkpoint["n_hidden_units"],
        )
        self.nn.load_state_dict(checkpoint["model_state_dict"])

        self.lr = checkpoint["optimizer_lr"]
        self.optimizer = optim.RMSprop(
            self.nn.parameters(), lr=self.lr, alpha=0.9, eps=1e-07
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def save_checkpoint(self, env, name: str = None) -> None:
        path = Path("models/")
        if not path.exists():
            path.mkdir()
        if name:
            model_path = path / f"{name}_{env.unwrapped.spec.id}.tar"
        else:
            date_string = datetime.now().strftime(r"%Y_%m_%d__%H_%M_%S")
            model_path = path / f"{date_string}_{env.unwrapped.spec.id}.tar"
        torch.save(
            {
                "env": env.unwrapped.spec.id,
                "env_state_dim": self.state_dim,
                "env_action_dim": self.action_dim,
                "n_hidden_layers": self.nn.n_hidden_layers,
                "n_hidden_units": self.nn.n_hidden_units,
                "model_state_dict": self.nn.state_dict(),
                "optimizer_lr": self.optimizer.param_groups[0]["lr"],
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            model_path,
        )


class A0CAgent(Agent):
    def __init__(
        self,
        Env: gym.Env,
        n_hidden_layers: int,
        n_hidden_units: int,
        value_loss_ratio: float,
        n_traces: int,
        lr: float,
        temperature: float,
        c_uct: float,
        c_pw: float,
        kappa: float,
        tau: float,
        alpha: float,
        gamma: float,
    ) -> None:
        # get info about the environment
        state_dim, self.state_discrete = check_space(Env.observation_space)
        self.state_dim = state_dim[0]
        action_dim, self.action_discrete = check_space(Env.action_space)
        self.action_dim = action_dim[0]

        # initialize values
        self.n_traces = n_traces
        self.c_uct = c_uct
        self.c_pw = c_pw
        self.kappa = kappa
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lr
        self.temperature = temperature
        self.value_loss_ratio = value_loss_ratio

        # action_dim*2 -> Needs both location and scale for one dimension
        self.nn = NetworkContinuous(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            act_limit=Env.action_space.high[0],
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
        )
        self.optimizer = optim.RMSprop(
            self.nn.parameters(), lr=self.lr, alpha=0.9, eps=1e-07
        )

    @property
    def n_hidden_layers(self) -> int:
        return self.nn.n_hidden_layers

    @property
    def n_hidden_units(self) -> int:
        return self.nn.n_hidden_units

    def reset_mcts(self, root_state: np.array) -> None:
        self.mcts = MCTSContinuous(
            model=self.nn,
            num_actions=self.nn.action_dim,
            gamma=self.gamma,
            c_uct=self.c_uct,
            c_pw=self.c_pw,
            kappa=self.kappa,
            root_state=root_state,
        )

    def act(
        self, Env: gym.Env, mcts_env: gym.Env, deterministic: bool = False
    ) -> Tuple[int, np.array, np.array, np.array]:
        self.mcts.search(
            n_traces=self.n_traces, Env=Env, mcts_env=mcts_env, simulation=False
        )
        state, actions, log_probs, log_counts, V_hat, V_target = self.mcts.return_results()
        if deterministic:
            action = actions[log_counts.argmax()]
        else:
            pi = stable_normalizer(log_counts, self.temperature)
            action = np.random.choice(actions, size=(1,), p=pi)

        return action, state, log_probs, log_counts, V_hat, V_target

    def _calculate_policy_loss(
        self, log_probs: torch.Tensor, log_counts: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:

        with torch.no_grad():
            # calculate scaling term
            log_diff = log_probs - self.tau * log_counts

        # multiple with log_probs gradient
        policy_loss = torch.einsum("ni, ni -> n", log_diff, log_probs)

        if reduction == "mean":
            return policy_loss.mean()
        else:
            return policy_loss.sum()

    def _calculate_entropy_loss(
        self, log_probs: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        if reduction == "mean":
            return -self.alpha * log_probs.mean()
        else:
            return -self.alpha * log_probs.sum()

    def calculate_loss(
        self,
        log_probs: torch.Tensor,
        log_counts: torch.tensor,
        V: torch.Tensor,
        V_hat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        policy_loss = self._calculate_policy_loss(
            log_probs, log_counts, reduction="mean"
        )
        entropy_loss = self._calculate_entropy_loss(log_probs)
        value_loss = self.value_loss_ratio * F.mse_loss(V_hat, V, reduction="mean")
        loss = policy_loss + entropy_loss + value_loss
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
        }

    def update(
        self, obs: Tuple[np.array, torch.Tensor, np.array, np.array]
    ) -> Dict[str, float]:
        self.optimizer.zero_grad()

        states, log_probs, log_counts, V_hat, V_target = obs
        states_tensor = torch.from_numpy(states).float()
        log_counts_tensor = torch.from_numpy(log_counts).float()
        values_tensor = torch.from_numpy(V_target).float()
        # TODO: Implement this using only actions and states as input
        _, _, V_hat = self.nn(states_tensor)

        loss_dict = self.calculate_loss(
            log_probs, log_counts_tensor, values_tensor, V_hat
        )
        with torch.autograd.set_detect_anomaly(True):
            loss_dict["loss"].backward()
            self.optimizer.step()

        loss_dict["loss"] = loss_dict["loss"].detach().item()
        loss_dict["policy_loss"] = loss_dict["policy_loss"].detach().item()
        loss_dict["entropy_loss"] = loss_dict["entropy_loss"].detach().item()
        loss_dict["value_loss"] = loss_dict["value_loss"].detach().item()
        return loss_dict

    def train(self, buffer: ReplayBuffer) -> float:
        buffer.reshuffle()
        running_loss = {"loss": 0, "policy_loss": 0, "entropy_loss": 0, "value_loss": 0}
        for epoch in range(1):
            for batches, obs in enumerate(buffer):
                loss = self.update(obs)
                running_loss["loss"] += loss["loss"]
                running_loss["policy_loss"] += loss["policy_loss"]
                running_loss["entropy_loss"] += loss["entropy_loss"]
                running_loss["value_loss"] += loss["value_loss"]
        for val in running_loss.values():
            val /= batches + 1
        return running_loss

    # TODO: Need to map location to device
    # self.policy.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    def load_checkpoint(self, name: str) -> None:
        model_path = (Path("models") / name).with_suffix(".tar")
        checkpoint = torch.load(model_path)
        self.nn = NetworkDiscrete(
            state_dim=checkpoint["env_state_dim"],
            action_dim=checkpoint["env_action_dim"],
            n_hidden_layers=checkpoint["n_hidden_layers"],
            n_hidden_units=checkpoint["n_hidden_units"],
        )
        self.nn.load_state_dict(checkpoint["model_state_dict"])

        self.lr = checkpoint["optimizer_lr"]
        self.optimizer = optim.RMSprop(
            self.nn.parameters(), lr=self.lr, alpha=0.9, eps=1e-07
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def save_checkpoint(self, env, name: str = None) -> None:
        path = Path("models/")
        if not path.exists():
            path.mkdir()
        if name:
            model_path = path / f"{name}_{env.unwrapped.spec.id}.tar"
        else:
            date_string = datetime.now().strftime(r"%Y_%m_%d__%H_%M_%S")
            model_path = path / f"{date_string}_{env.unwrapped.spec.id}.tar"
        torch.save(
            {
                "env": env.unwrapped.spec.id,
                "env_state_dim": self.state_dim[0],
                "env_action_dim": self.action_dim,
                "n_hidden_layers": self.nn.n_hidden_layers,
                "n_hidden_units": self.nn.n_hidden_units,
                "model_state_dict": self.nn.state_dict(),
                "optimizer_lr": self.optimizer.param_groups[0]["lr"],
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            model_path,
        )
