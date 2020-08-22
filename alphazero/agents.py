import torch
import torch.nn.functional as F
from torch import optim
from pathlib import Path
from datetime import datetime

from .networks import Network
from .alphazero import MCTS
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
    ):

        # initialize values
        self.n_traces = n_traces
        self.c_uct = c_uct
        self.gamma = gamma
        self.lr = lr
        self.temperature = temperature

        self.is_atari = is_atari_game(Env)

        self.nn = Network(
            Env, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units
        )
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
        states_tensor = torch.from_numpy(state_batch).float()
        values_tensor = torch.from_numpy(V_batch).float()
        action_probs_tensor = torch.from_numpy(pi_batch).float()

        pi_logits, V_hat = self.nn(states_tensor)
        loss = self.calculate_loss(pi_logits, V_hat, values_tensor, action_probs_tensor)
        loss.backward()
        self.optimizer.step()

        return loss.detach().item()

    def train(self, buffer):
        buffer.reshuffle()
        running_loss = []
        for epoch in range(1):
            for batches, obs in enumerate(buffer):
                loss = self.update(obs)
                running_loss.append(loss)
        episode_loss = sum(running_loss) / (batches + 1)
        return episode_loss

    # TODO: Add loading ptions
    def load_model(self):
        pass

    def save_checkpoint(self, env):
        path = Path("models/")
        if not path.exists():
            path.mkdir()

        model_path = path/f"{datetime.now()}_{env.unwrapped.spec.id}_.pt"
        torch.save({
            "env": env.unwrapped.spec.id,
            "env_state_dim": self.nn.state_dim[0],
            "env_action_dim": self.nn.action_dim,
            "n_hidden_layers" : self.nn.n_hidden_layers,
            "n_hidden_units": self.nn.n_hidden_units,
            "model_state_dict": self.nn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            }, model_path)
        torch.save(self.nn.state_dict(), model_path)

