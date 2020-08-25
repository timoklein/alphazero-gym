import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import gym
from pathlib import Path
from datetime import datetime
from typing import Tuple

from torch.optim.rmsprop import RMSprop

from .networks import NetworkDiscrete
from .alphazero import MCTS,Action, Node
from .helpers import is_atari_game, check_space
from .buffers import ReplayBuffer

class AlphaZeroAgent:
    def __init__(
        self,
        Env: gym.Env,
        n_hidden_layers: int,
        n_hidden_units: int,
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

        self.is_atari = is_atari_game(Env)

        self.nn = NetworkDiscrete(
            self.state_dim[0], self.action_dim, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units
        )
        self.optimizer = optim.RMSprop(
            self.nn.parameters(), lr=self.lr, alpha=0.9, eps=1e-07
        )

    def reset_mcts(self, root_state: np.array) -> None:
        self.mcts = MCTS(
            model=self.nn,
            num_actions=self.nn.action_dim,
            is_atari=self.is_atari,
            gamma=self.gamma,
            c_uct=self.c_uct,
            root_state=root_state,
        )

    def search(self, Env: gym.Env, mcts_env: gym.Env):
        self.mcts.search(n_traces=self.n_traces, Env=Env, mcts_env=mcts_env)
        state, pi, V = self.mcts.return_results(self.temperature)
        return state, pi, V

    def mcts_forward(self, action: Action, node: Node):
        self.mcts.forward(action, node)

    @staticmethod
    def calculate_loss(pi_logits: torch.Tensor, V_hat: torch.tensor, V, pi: torch.Tensor, value_ratio: float=1):
        # calculate policy loss from model logits
        # first we have to convert the probabilities to labels
        pi = pi.argmax(dim=1)
        pi_loss = F.cross_entropy(pi_logits, pi)
        # value loss
        v_loss = F.mse_loss(V_hat, V)
        loss = pi_loss + value_ratio * v_loss
        return loss

    def update(self, obs: Tuple[np.array, np.array, np.array]) -> float:
        self.optimizer.zero_grad()

        state_batch, V_batch, pi_batch = obs
        states_tensor = torch.from_numpy(state_batch).float()
        values_tensor = torch.from_numpy(V_batch).float()
        action_probs_tensor = torch.from_numpy(pi_batch).float()

        pi_logits, V_hat = self.nn(states_tensor)
        loss = self.calculate_loss(pi_logits, V_hat, values_tensor, action_probs_tensor)
        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def train(self, buffer: ReplayBuffer) -> float:
        buffer.reshuffle()
        running_loss = []
        for epoch in range(1):
            for batches, obs in enumerate(buffer):
                loss = self.update(obs)
                running_loss.append(loss)
        episode_loss = sum(running_loss) / (batches + 1)
        return episode_loss

    # TODO: Need to map location to device
    # self.policy.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    def load_checkpoint(self, name: str) -> None:
        model_path = (Path("models")/name).with_suffix(".tar")
        checkpoint = torch.load(model_path)
        self.nn = NetworkDiscrete(
            state_dim=checkpoint["env_state_dim"], action_dim=checkpoint["env_action_dim"],
            n_hidden_layers=checkpoint["n_hidden_layers"], n_hidden_units=checkpoint["n_hidden_units"]
        )
        self.nn.load_state_dict(checkpoint["model_state_dict"])

        self.lr = checkpoint["optimizer_lr"]
        self.optimizer = optim.RMSprop(
            self.nn.parameters(), lr=self.lr, alpha=0.9, eps=1e-07
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def save_checkpoint(self, env, name: str=None) -> None:
        path = Path("models/")
        if not path.exists():
            path.mkdir()
        if name:
            model_path = path/f"{name}_{env.unwrapped.spec.id}.tar"
        else:
            date_string = datetime.now().strftime(r"%Y_%m_%d__%H_%M_%S")
            model_path = path/f"{date_string}_{env.unwrapped.spec.id}.tar"
        torch.save({
            "env": env.unwrapped.spec.id,
            "env_state_dim": self.state_dim[0],
            "env_action_dim": self.action_dim,
            "n_hidden_layers": self.nn.n_hidden_layers,
            "n_hidden_units": self.nn.n_hidden_units,
            "model_state_dict": self.nn.state_dict(),
            "optimizer_lr": self.optimizer.param_groups[0]['lr'],
            "optimizer_state_dict": self.optimizer.state_dict(),
            }, model_path)

