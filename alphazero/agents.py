from warnings import warn
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim

from.networks import Network
from .mcts import MCTS
from .helpers import is_atari_game


class AlphaZeroAgent:
    def __init__(
        self,
        Env,
        n_hidden_layers,
        n_hidden_units,
        n_traces,
        lr,
        temperature,
        c_uct,
        gamma,
        device
    ):

        # initialize values
        self.n_traces = n_traces
        self.c_uct = c_uct
        self.gamma = gamma
        self.lr = lr
        self.temperature = temperature

        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device.type == "cpu":
                warn("No GPU available. Training on CPU despice device being set to GPU.", RuntimeWarning)
        else:
            self.device = torch.device("cpu")

        self.is_atari = is_atari_game(Env)

        self.nn = Network(
            Env, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units
        ).to(self.device)
        self.optimizer = optim.RMSprop(
            self.nn.parameters(), lr=self.lr, alpha=0.9, eps=1e-07
        )

    def reset_mcts(self, root_state):
        self.mcts = MCTS(
            model=self.nn,
            num_actions=self.nn.action_dim,
            is_atari=self.is_atari,
            gamma=self.gamma,
            c_uct=self.c_uct,
            root_state=root_state,
        )

    def search(self, Env, mcts_env):
        self.mcts.search(n_traces=self.n_traces, Env=Env, mcts_env=mcts_env)
        state, pi, V = self.mcts.return_results(self.temperature)
        return state, pi, V

    def mcts_forward(self, action, node):
        self.mcts.forward(action, node)

    @staticmethod
    def calculate_loss(pi_logits, V_hat, V, pi, value_ratio=1):
        # calculate policy loss from model logits
        # first we have to convert the probabilities to labels
        pi = pi.argmax(dim=1)
        pi_loss = F.cross_entropy(pi_logits, pi)
        # value loss
        v_loss = F.mse_loss(V_hat, V)
        loss = pi_loss + value_ratio * v_loss
        return loss

    def update(self, obs):
        self.optimizer.zero_grad()

        state_batch, V_batch, pi_batch = obs
        states_tensor = torch.from_numpy(state_batch).float().to(self.device)
        values_tensor = torch.from_numpy(V_batch).float().to(self.device)
        action_probs_tensor = torch.from_numpy(pi_batch).float().to(self.device)

        pi_logits, V_hat = self.nn(states_tensor)
        loss = self.calculate_loss(pi_logits, V_hat, values_tensor, action_probs_tensor)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def train(self, buffer):
        buffer.reshuffle()
        running_loss = []
        for epoch in range(1):
            for batches, obs in enumerate(buffer):
                loss = self.update(obs)
                running_loss.append(loss)
        episode_loss = sum(running_loss) / (batches + 1)
        return episode_loss

    def load_model(self, state_dict_path):
        print(f"Loading models from {state_dict_path}.")
        self.policy.load_state_dict(torch.load(state_dict_path, map_location=self.device))

    def save_model(self, env_name: str, suffix: str = ".pt"):
        path = Path("models/")
        if not path.exists():
            path.mkdir()
        nn_path = (path/f"{datetime.now()}_{env_name}_AZ_discrete").with_suffix(suffix)
        print(f"Saving network to {nn_path}.")
        torch.save(self.nn.state_dict(), nn_path)