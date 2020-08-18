#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-player Alpha Zero
@author: Thomas Moerland, Delft University of Technology
"""
import copy
from os import stat
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

from rl.make_game import make_game
from helpers import (
    check_space,
    is_atari_game,
    copy_atari_state,
    restore_atari_state,
    stable_normalizer,
    argmax,
)


#### Neural Networks ##
# TODO: Add variable number of layers
class Network(nn.Module):
    def __init__(self, Env, n_hidden_layers, n_hidden_units):
        super().__init__()

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units

        self.state_dim, self.state_discrete = check_space(Env.observation_space)
        self.action_dim, self.action_discrete = check_space(Env.action_space)
        self.in_layer = nn.Linear(self.state_dim[0], n_hidden_units)
        self.hidden = nn.Linear(n_hidden_units, n_hidden_units)
        self.policy_head = nn.Linear(n_hidden_units, self.action_dim)
        self.value_head = nn.Linear(n_hidden_units, 1)

    def forward(self, x):
        x = F.elu(self.in_layer(x))
        x = F.elu(self.hidden(x))
        # no need for softmax, can be computed directly from cross-entropy loss
        pi_hat = self.policy_head(x)
        V_hat = self.value_head(x)
        return pi_hat, V_hat

    @torch.no_grad()
    def predict_V(self, x):
        x = F.elu(self.in_layer(x))
        x = F.elu(self.hidden(x))
        V_hat = self.value_head(x)
        return V_hat.detach().cpu().numpy()

    @torch.no_grad()
    def predict_pi(self, x):
        x = F.elu(self.in_layer(x))
        x = F.elu(self.hidden(x))
        pi_hat = F.softmax(self.policy_head(x), dim=-1)
        return pi_hat.detach().cpu().numpy()


##### MCTS functions #####
class Action:
    """ Action object """

    def __init__(self, index, parent_state, Q_init=0.0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 0
        self.Q = Q_init

    def add_child_state(self, s1, r, terminal, model):
        self.child_state = State(s1, r, terminal, self, self.parent_state.na, model)
        return self.child_state

    def update(self, R):
        self.n += 1
        self.W += R
        self.Q = self.W / self.n


class State:
    """ State object """

    def __init__(self, index, r, terminal, parent_action, na, model):
        """ Initialize a new state """
        self.index = index  # state
        self.r = r  # reward upon arriving in this state
        self.terminal = terminal  # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 0
        self.model = model

        self.evaluate()
        # Child actions
        self.na = na
        self.child_actions = [
            Action(a, parent_state=self, Q_init=self.V) for a in range(na)
        ]
        state = torch.from_numpy(index[None,]).float()
        self.priors = model.predict_pi(state).flatten()

    def select(self, c=1.5):
        """ Select one of the child actions based on UCT rule """
        UCT = np.array(
            [
                child_action.Q
                + prior * c * (np.sqrt(self.n + 1) / (child_action.n + 1))
                for child_action, prior in zip(self.child_actions, self.priors)
            ]
        )
        winner = argmax(UCT)
        return self.child_actions[winner]

    def evaluate(self):
        """ Bootstrap the state value """
        state = torch.from_numpy(self.index[None,]).float()
        self.V = (
            np.squeeze(self.model.predict_V(state))
            if not self.terminal
            else np.array(0.0)
        )

    def update(self):
        """ update count on backward pass """
        self.n += 1


class MCTS:
    """ MCTS object """

    def __init__(self, root, root_index, model, na, gamma):
        self.root = None
        self.root_index = root_index
        self.model = model
        self.na = na
        self.gamma = gamma

    def search(self, n_traces, c, Env, mcts_env):
        """ Perform the MCTS search from the root """
        if self.root is None:
            self.root = State(
                self.root_index,
                r=0.0,
                terminal=False,
                parent_action=None,
                na=self.na,
                model=self.model,
            )  # initialize new root
        else:
            self.root.parent_action = None  # continue from current root
        if self.root.terminal:
            raise ValueError("Can't do tree search from a terminal state")

        is_atari = is_atari_game(Env)
        if is_atari:
            snapshot = copy_atari_state(
                Env
            )  # for Atari: snapshot the root at the beginning

        for i in range(n_traces):
            state = self.root  # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(Env)  # copy original Env to rollout from
            else:
                restore_atari_state(mcts_env, snapshot)

            while not state.terminal:
                action = state.select(c=c)
                s1, r, t, _ = mcts_env.step(action.index)
                if hasattr(action, "child_state"):
                    state = action.child_state  # select
                    continue
                else:
                    state = action.add_child_state(s1, r, t, self.model)  # expand
                    break

            # Back-up
            R = state.V
            while state.parent_action is not None:  # loop back-up until root is reached
                R = state.r + self.gamma * R
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()

    def return_results(self, temp):
        """ Process the output at the root node """
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        pi_target = stable_normalizer(counts, temp)
        V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        return self.root.index, pi_target, V_target

    def forward(self, action, state):
        """ Move the root forward """
        if not hasattr(self.root.child_actions[action], "child_state"):
            self.root = None
            self.root_index = state
        elif np.linalg.norm(self.root.child_actions[action].child_state.index - state) > 0.01:
            print(
                "Warning: this domain seems stochastic. Not re-using the subtree for next search. "
                + "To deal with stochastic environments, implement progressive widening."
            )
            self.root = None
            self.root_index = state
        else:
            self.root = self.root.child_actions[action].child_state

# TODO: Add GPU Training
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

        self.nn = Network(Env, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units)
        self.optimizer = optim.RMSprop(
            self.nn.parameters(), lr=self.lr, alpha=0.9, eps=1e-07
        )

    def reset_mcts(self, root_index):
        self.mcts = MCTS(
            root_index=root_index,
            root=None,
            model=self.nn,
            na=self.nn.action_dim,
            gamma=self.gamma,
        )

    def search(self, Env, mcts_env):
        self.mcts.search(
            n_traces=self.n_traces, c=self.c_uct, Env=Env, mcts_env=mcts_env
        )
        state, pi, V = self.mcts.return_results(self.temperature)
        return state, pi, V

    def mcts_forward(self, action, state):
        self.mcts.forward(action, state)

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

    # TODO: Add loading and saving options
    def load_model(self):
        pass

    def save_model(self):
        pass


if __name__ == "__main__":
    Env = make_game("CartPole-v0")
    model = Network(Env, 128)
    state_dim, state_discrete = check_space(Env.observation_space)
    action_dim, action_discrete = check_space(Env.action_space)
    print(f"State dimension: {state_dim}, discrete={state_discrete}.")
    print(f"Action dimension: {action_dim}, discrete={action_discrete}.")

    test = torch.randn(32, 4)
    pi, V = model(test)
    print(pi, V)
