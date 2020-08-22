import torch
from torch import nn
import torch.nn.functional as F

from .helpers import check_space

class Network(nn.Module):
    def __init__(self, Env, n_hidden_layers, n_hidden_units):
        super().__init__()

        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units

        layers = []
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(n_hidden_units, n_hidden_units))
            layers.append(nn.ELU())

        self.state_dim, self.state_discrete = check_space(Env.observation_space)
        self.action_dim, self.action_discrete = check_space(Env.action_space)
        self.in_layer = nn.Linear(self.state_dim[0], n_hidden_units)
        
        self.hidden = nn.Sequential(*layers)

        self.policy_head = nn.Linear(n_hidden_units, self.action_dim)
        self.value_head = nn.Linear(n_hidden_units, 1)

    def forward(self, x):
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        # no need for softmax, can be computed directly from cross-entropy loss
        pi_hat = self.policy_head(x)
        V_hat = self.value_head(x)
        return pi_hat, V_hat

    @torch.no_grad()
    def predict_V(self, x):
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        V_hat = self.value_head(x)
        return V_hat.detach().cpu().numpy()

    @torch.no_grad()
    def predict_pi(self, x):
        x = F.elu(self.in_layer(x))
        x = self.hidden(x)
        pi_hat = F.softmax(self.policy_head(x), dim=-1)
        return pi_hat.detach().cpu().numpy()