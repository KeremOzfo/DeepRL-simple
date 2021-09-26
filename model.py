import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

from typing import Dict, Optional, cast

import numpy as np
import torch
from torch import distributions
import utils as ptu
from typing import Union


Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)

class Agent(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline
        self.baseline_loss = nn.MSELoss()

        if self.discrete: # Discrete action space
            self.logits_na = build_mlp(input_size=self.ob_dim,# NN architecture for taking action
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else: # Continous action space
            self.logits_na = None
            self.mean_net = build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size) # NN for finding state based mean for the continuous action
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline: # NN architecture for baseline
            self.baseline = build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray: # This is part of typing module for type checking see: https://www.journaldev.com/34519/python-typing-module
        # Obs should be np.ndarray and function should return np.ndarray

        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None] # This works identical to the unsuqeeze method for tensor i.e., adds one more dimension

        observation_tensor = torch.tensor(observation, dtype=torch.float).to(ptu.device)
        action_distribution = self.forward(observation_tensor) # Gives the action distribution based on observation
        return cast(
            np.ndarray,
            action_distribution.sample().cpu().detach().numpy(), # .numpy() ; tensor -> np.ndarray
        )


    # update/train this policy
    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        ################ We want to MAXIMIZE ######################
        # is the expectation over collected trajectories of:
        # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]

        actions_distribution = self.forward(observations)
        # Annotates log_probs is a torch.Tensor
        log_probs: torch.Tensor = actions_distribution.log_prob(actions)
        if not self.discrete:  # Continuous
            log_probs = log_probs.sum(1)
        assert log_probs.size() == advantages.size()
        loss = -(log_probs * advantages).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.nn_baseline:
            # Normalize the q values
            targets = ptu.normalize(q_values, q_values.mean(), q_values.std())  # This is the real observation
            targets = ptu.from_numpy(targets)
            baseline_predictions: torch.Tensor = self.baseline(
                observations).squeeze()  # This is the estimate of baseline NN

            ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
            ## [ N ] versus shape [ N x 1 ]
            ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1

            assert baseline_predictions.shape == targets.shape

            ####### Use mean MSE loss for baseline ########
            baseline_loss = F.mse_loss(baseline_predictions, targets)

            ####### Train baseline network
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

        train_log = {
            'Training Loss': ptu.to_numpy(loss),
        }
        return train_log

    def run_baseline_prediction(self, obs):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> distributions.Distribution: # The function return Distribution object

        if self.discrete:
            return distributions.Categorical(logits=self.logits_na(observation)) # Action probabilities based on logit values
            # https://pytorch.org/docs/stable/distributions.html
            # Creates a categorical distribution parameterized by either probs or logits
        else:
            assert self.logstd is not None
            # Generate random continious action based on observation based mean and given std
            return distributions.Normal(self.mean_net(observation),torch.exp(self.logstd)[None]) #??????????????????? Why we use [None] is the action multi dimensional ?
            # Creates a normal (also called Gaussian) distribution parameterized by loc (mean) and scale (std).
