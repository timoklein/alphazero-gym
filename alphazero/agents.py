import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import gym
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from torch.optim.rmsprop import RMSprop

from .networks import NetworkDiscrete
from .alphazero import NNMCTSDiscrete, ActionDiscrete, NodeDiscrete
from .helpers import is_atari_game, check_space
from .buffers import ReplayBuffer


class AlphaZeroAgent:
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
        self.state_dim, self.state_discrete = check_space(Env.observation_space)
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
            self.state_dim[0],
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
        self.mcts = NNMCTSDiscrete(
            model=self.nn,
            num_actions=self.nn.action_dim,
            is_atari=self.is_atari,
            gamma=self.gamma,
            c_uct=self.c_uct,
            root_state=root_state,
        )

    def act(
        self, Env: gym.Env, mcts_env: gym.Env, deterministic: bool = False
    ) -> Tuple[int, np.array, np.array, float]:
        self.mcts.search(n_traces=self.n_traces, Env=Env, mcts_env=mcts_env)
        state, pi, V = self.mcts.return_results(self.temperature)
        # sample an action from the policy or pick best action if deterministic
        action = pi.argmax() if deterministic else np.random.choice(len(pi), p=pi)
        return action, state, pi, V

    def mcts_forward(self, action: ActionDiscrete, node: NodeDiscrete) -> None:
        self.mcts.forward(action, node)

    def calculate_loss(
        self, pi_logits: torch.Tensor, V_hat: torch.tensor, V, pi: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # calculate policy loss from model logits
        # first we have to convert the probabilities to labels
        pi = pi.argmax(dim=1)
        pi_loss = F.cross_entropy(pi_logits, pi)
        # value loss
        v_loss = F.mse_loss(V_hat, V)
        loss = pi_loss + self.value_loss_ratio * v_loss
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

