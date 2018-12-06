import torch
import torch.nn as nn
import torch.nn.functional as F


class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussian):
        super(MDN, self).__init__()
        self.hidden = nn.Linear(1, n_hidden)
        self.tanh = nn.Tanh()

        self.z_pi = nn.Linear(n_hidden, n_gaussian)
        self.z_mu = nn.Linear(n_hidden, n_gaussian)
        self.z_sigma = nn.Linear(n_hidden, n_gaussian)

    def forward(self, x):
        z_h = self.tanh(self.hidden(x))
        pi = F.softmax(self.z_pi(z_h), -1)
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))

        return pi, mu, sigma
