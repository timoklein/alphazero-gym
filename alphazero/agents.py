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
    ) -> None:

        self.nn = hydra.utils.instantiate(network_cfg)
        self.mcts = hydra.utils.instantiate(mcts_cfg, model=self.nn)
        self.loss = hydra.utils.instantiate(loss_cfg)
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

    @abstractmethod
    def load_checkpoint(self):
        ...

    @abstractmethod
    def save_checkpoint(self):
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
    ) -> None:

        super().__init__(
            network_cfg=network_cfg,
            loss_cfg=loss_cfg,
            mcts_cfg=mcts_cfg,
            optimizer_cfg=optimizer_cfg,
            grad_clip=grad_clip,
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
        self.optimizer.zero_grad()

        states, actions, counts, V_target = obs
        states_tensor = torch.from_numpy(states).float()
        values_tensor = torch.from_numpy(V_target).unsqueeze(dim=1).float()

        action_probs_tensor = F.softmax(torch.from_numpy(counts).float(), dim=-1)

        # TODO: Needs to work for both losses
        pi_logits, V_hat = self.nn(states_tensor)
        loss_dict = self.loss(pi_logits, action_probs_tensor, V_hat, values_tensor)
        loss_dict["loss"].backward()

        if self.clip:
            clip_grad_norm(self.nn.parameters(), self.clip)

        self.optimizer.step()

        info_dict = {key: float(value) for key, value in loss_dict.items()}
        return info_dict

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
        # self.value_loss_ratio = checkpoint["agent"]["value_loss_ratio"]

    # TODO: Load the loss correctly
    # TODO: Fix loss saving and checkpointing
    def save_checkpoint(self, env, name: str = None) -> None:
        path = Path("models/")
        if not path.exists():
            path.mkdir()
        if name:
            model_path = path / f"{name}_{env.unwrapped.spec.id}.tar"
        else:
            date_string = datetime.now().strftime(r"%Y_%m_%d__%H_%M_%S")
            model_path = path / f"{date_string}_{env.unwrapped.spec.id}.tar"
        loss_info = self.loss.get_info()
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
                },
                "loss": loss_info,
            },
            model_path,
        )


class ContinuousAgent(Agent):
    def __init__(
        self,
        network_cfg: DictConfig,
        mcts_cfg: DictConfig,
        loss_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        grad_clip: float,
    ) -> None:

        super().__init__(
            network_cfg=network_cfg,
            loss_cfg=loss_cfg,
            mcts_cfg=mcts_cfg,
            optimizer_cfg=optimizer_cfg,
            grad_clip=grad_clip,
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
        self.optimizer.zero_grad()

        states, actions, counts, V_target = obs

        actions_tensor = torch.from_numpy(actions).float()
        states_tensor = torch.from_numpy(states).float()
        values_tensor = torch.from_numpy(V_target).unsqueeze(dim=1).float()

        log_counts_tensor = torch.log(torch.from_numpy(counts).float())

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

    # TODO: Load the loss correctly
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
        self.optimizer = RMSprop(self.nn.parameters(), lr=self.lr, alpha=0.9, eps=1e-07)
        self.optimizer.load_state_dict(checkpoint["network_optimizer_state_dict"])

        self.n_traces = checkpoint["agent"]["n_traces"]
        self.c_uct = checkpoint["agent"]["c_uct"]
        self.c_pw = checkpoint["agent"]["c_pw"]
        self.kappa = checkpoint["agent"]["kappa"]
        # self.tau = checkpoint["agent"]["tau"]
        self.alpha = checkpoint["agent"]["alpha"]
        self.gamma = checkpoint["agent"]["gamma"]
        # self.value_loss_ratio = checkpoint["agent"]["value_loss_ratio"]

    def save_checkpoint(self, env, name: str = None) -> None:
        path = Path("models/")
        if not path.exists():
            path.mkdir()
        if name:
            model_path = path / f"{name}_{env.unwrapped.spec.id}.tar"
        else:
            date_string = datetime.now().strftime(r"%Y_%m_%d__%H_%M_%S")
            model_path = path / f"{date_string}_{env.unwrapped.spec.id}.tar"

        loss_info = {"name": type(self.loss).__name__}
        loss_info.update(
            {
                key: getattr(self.loss, key)
                for key in vars(self.loss)
                if not key.startswith("_") and not key.startswith("training")
            }
        )

        torch.save(
            {
                "env": env.unwrapped.spec.id,
                "env_state_dim": self.state_dim,
                "env_action_dim": self.action_dim,
                "act_limit": env.action_space.high[0],
                "n_hidden_layers": self.nn.n_hidden_layers,
                "n_hidden_units": self.nn.n_hidden_units,
                "model_state_dict": self.nn.state_dict(),
                "network_optimizer_lr": self.optimizer.param_groups[0]["lr"],
                "network_optimizer_state_dict": self.optimizer.state_dict(),
                "agent": {
                    "n_traces": self.n_traces,
                    "c_uct": self.c_uct,
                    "c_pw": self.c_pw,
                    "kappa": self.kappa,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                },
            },
            model_path,
        )
