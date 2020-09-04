import torch
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
import numpy as np
import gym
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from abc import ABC, abstractmethod

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


# TODO: Add num_training epochs parameter


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
        self.optimizer = RMSprop(self.nn.parameters(), lr=self.lr, alpha=0.9, eps=1e-07)

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
        self.optimizer = RMSprop(self.nn.parameters(), lr=self.lr, alpha=0.9, eps=1e-07)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.n_traces = checkpoint["agent"]["n_traces"]
        self.c_uct = checkpoint["agent"]["c_uct"]
        self.gamma = checkpoint["agent"]["gamma"]
        self.temperature = checkpoint["agent"]["temperature"]
        self.value_loss_ratio = checkpoint["agent"]["value_loss_ratio"]

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
                "agent": {
                    "n_traces": self.n_traces,
                    "c_uct": self.c_uct,
                    "gamma": self.gamma,
                    "temperature": self.temperature,
                    "value_loss_ratio": self.value_loss_ratio,
                },
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

        self.autotune = True if alpha is None else False

        # initialize values
        self.n_traces = n_traces
        self.c_uct = c_uct
        self.c_pw = c_pw
        self.kappa = kappa
        self.tau = tau
        self.gamma = gamma
        self.lr = lr
        self.value_loss_ratio = value_loss_ratio

        if self.autotune:
            # set target entropy to -|A|
            self.target_entropy = -self.action_dim
            # initialize alpha to 1
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp().item()
            # for simplicity: Use the same optimizer settings as for the neural network
            self.a_optimizer = Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = alpha

        # action_dim*2 -> Needs both location and scale for one dimension
        self.nn = NetworkContinuous(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            act_limit=Env.action_space.high[0],
            n_hidden_layers=n_hidden_layers,
            n_hidden_units=n_hidden_units,
        )
        self.n_optimizer = RMSprop(
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

    # TODO: Pass deterministic acting to the neural network instead of using it like this
    def act(
        self, Env: gym.Env, mcts_env: gym.Env, deterministic: bool = True
    ) -> Tuple[int, np.array, np.array, np.array]:
        self.mcts.search(
            n_traces=self.n_traces, Env=Env, mcts_env=mcts_env, simulation=False
        )
        state, actions, log_counts, V_hat = self.mcts.return_results()

        # select the action that was visited most
        action = actions[log_counts.argmax()][np.newaxis]

        return action, actions, state, log_counts, V_hat

    def _calculate_policy_loss(
        self, log_probs: torch.Tensor, log_counts: torch.Tensor, reduction: str
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

    def calculate_loss(
        self,
        log_probs: torch.Tensor,
        log_counts: torch.tensor,
        entropy: torch.Tensor,
        V: torch.Tensor,
        V_hat: torch.Tensor,
        reduce: str = "mean",
    ) -> Dict[str, torch.Tensor]:

        policy_loss = self._calculate_policy_loss(
            log_probs, log_counts, reduction=reduce
        )
        entropy_loss = entropy.mean() if reduce == "mean" else entropy
        entropy_loss *= -self.alpha
        value_loss = self.value_loss_ratio * F.mse_loss(V_hat, V, reduction=reduce)
        loss = policy_loss + entropy_loss + value_loss
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
        }

    def update(
        self, obs: Tuple[np.array, np.array, np.array, np.array]
    ) -> Dict[str, float]:
        self.n_optimizer.zero_grad()

        actions, states, log_counts, V_target = obs

        actions_tensor = torch.from_numpy(actions).float()
        states_tensor = torch.from_numpy(states).float()
        log_counts_tensor = torch.from_numpy(log_counts).float()
        values_tensor = torch.from_numpy(V_target).unsqueeze(dim=1).float()
        log_probs, entropy, V_hat = self.nn.get_train_data(
            states_tensor, actions_tensor
        )

        loss_dict = self.calculate_loss(
            log_probs=log_probs,
            log_counts=log_counts_tensor,
            entropy=entropy,
            V=values_tensor,
            V_hat=V_hat,
        )
        loss_dict["loss"].backward()
        self.n_optimizer.step()

        if self.autotune:
            # we don't want to backprop through the network here
            a_entropy = entropy.detach()
            # calculate loss for entropy regularization parameter
            alpha_loss = (-self.log_alpha * (a_entropy + self.target_entropy)).mean()
            # optimize and set values
            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
            loss_dict["alpha_loss"] = alpha_loss.detach().item()

        loss_dict["loss"] = loss_dict["loss"].detach().item()
        loss_dict["policy_loss"] = loss_dict["policy_loss"].detach().item()
        loss_dict["entropy_loss"] = loss_dict["entropy_loss"].detach().item()
        loss_dict["value_loss"] = loss_dict["value_loss"].detach().item()
        return loss_dict

    def train(self, buffer: ReplayBuffer) -> float:
        buffer.reshuffle()
        running_loss = defaultdict(float)
        for epoch in range(1):
            for batches, obs in enumerate(buffer):
                loss = self.update(obs)
                running_loss["loss"] += loss["loss"]
                running_loss["policy_loss"] += loss["policy_loss"]
                running_loss["entropy_loss"] += loss["entropy_loss"]
                running_loss["value_loss"] += loss["value_loss"]
                if self.autotune:
                    running_loss["alpha_loss"] += loss["alpha_loss"]
        for val in running_loss.values():
            val /= batches + 1
        return running_loss

    # TODO: Need to map location to device
    # self.policy.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    def load_checkpoint(self, name: str) -> None:
        model_path = (Path("models") / name).with_suffix(".tar")
        checkpoint = torch.load(model_path)
        self.nn = NetworkContinuous(
            state_dim=checkpoint["env_state_dim"],
            action_dim=checkpoint["env_action_dim"],
            act_limit=checkpoint["act_limit"],
            n_hidden_layers=checkpoint["n_hidden_layers"],
            n_hidden_units=checkpoint["n_hidden_units"],
        )
        self.nn.load_state_dict(checkpoint["model_state_dict"])

        self.lr = checkpoint["network_optimizer_lr"]
        self.n_optimizer = RMSprop(
            self.nn.parameters(), lr=self.lr, alpha=0.9, eps=1e-07
        )
        self.n_optimizer.load_state_dict(checkpoint["network_optimizer_state_dict"])

        self.n_traces = checkpoint["agent"]["n_traces"]
        self.c_uct = checkpoint["agent"]["c_uct"]
        self.c_pw = checkpoint["agent"]["c_pw"]
        self.kappa = checkpoint["agent"]["kappa"]
        self.tau = checkpoint["agent"]["tau"]
        self.alpha = checkpoint["agent"]["alpha"]
        self.gamma = checkpoint["agent"]["gamma"]
        self.value_loss_ratio = checkpoint["agent"]["value_loss_ratio"]

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
                "act_limit": env.action_space.high[0],
                "n_hidden_layers": self.nn.n_hidden_layers,
                "n_hidden_units": self.nn.n_hidden_units,
                "model_state_dict": self.nn.state_dict(),
                "network_optimizer_lr": self.n_optimizer.param_groups[0]["lr"],
                "network_optimizer_state_dict": self.n_optimizer.state_dict(),
                "agent": {
                    "n_traces": self.n_traces,
                    "c_uct": self.c_uct,
                    "c_pw": self.c_pw,
                    "kappa": self.kappa,
                    "tau": self.tau,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "value_loss_ratio": self.value_loss_ratio,
                },
            },
            model_path,
        )
