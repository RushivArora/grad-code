import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.distributions import Categorical, Bernoulli

import matplotlib.pyplot as plt
import numpy as np
from math import exp
import random
from collections import deque
import os


class StateModel(nn.Module):
    def __init__(self, num_states, num_outputs):
        super(StateModel, self).__init__()
        self.num_states = num_states
        self.num_outputs = num_outputs

        """
        self.features = nn.Sequential(
            nn.Linear(num_states, 32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU()
        )
        """

        self.features = nn.Sequential(
            nn.Linear(num_states, num_outputs),
            nn.ReLU()
        )

    def forward(self, observation):
        state = self.features(observation)
        return state.detach()

"""

This is the Uniform Policy-over-Options

"""
class PolicyOverOptions(nn.Module):
    def __init__(self, rng, num_states, num_options, eps_start=1.0, eps_min=0.01, eps_decay=int(1e5)):
        super(PolicyOverOptions, self).__init__()
        self.rng = rng
        self.num_states = num_states
        self.num_options = num_options

        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.num_steps = 0

        self.linear = nn.Linear(self.num_states, self.num_options)
        self.tanh = nn.Tanh()
        self.soft = nn.Softmax(dim=0)

    def Q_omega(self, state, option=None):
        out = self.linear(state)
        return self.soft(out)

    def sample_option(self, state):
        epsilon = self.epsilon
        probs = self.linear(state)
        e_greedy_option = self.soft(probs).argmax(dim=-1).item()
        option = np.random.choice(self.num_options) if np.random.rand() < epsilon else e_greedy_option
        return option

    @property
    def epsilon(self):
        eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
        self.num_steps += 1
        # return eps
        return 0.01

"""

This is the Interest Policy-over-Options

"""
class InterestPolicyOverOptions(nn.Module):
    def __init__(self, rng, num_states, num_options, policy_over_options, k = 0., eps_start=1.0, eps_min=0.01,
                 eps_decay=int(1e5)):
        super(InterestPolicyOverOptions, self).__init__()
        self.rng = rng
        self.num_states = num_states
        self.num_options = num_options
        self.policy_over_options = policy_over_options
        self.linear =  nn.Linear(self.num_states, self.num_options)
        self.tanh = nn.Tanh()
        self.sig = torch.nn.Sigmoid()
        self.k = k
        self.inductive_bias()

        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.num_steps = 0

    def inductive_bias(self):
        return

    def Q_omega(self, state, option=None):
        out = self.linear(state)
        return self.sig(out)

    def get_interest(self, state, option):
        out = self.linear(state)
        return self.sig(out)[option]

    def sample_option(self, obs):
        op_prob = self.policy_over_options.Q_omega(obs)
        int_func = self.linear(obs)
        int_func = self.sig(int_func)
        #print(int_func)
        activated_options = []
        for int_val in int_func:
            if int_val >= self.k:
                activated_options.append(1.)
            else:
                activated_options.append(0.)
        indices = (-int_func).argsort()[:2]
        if 1. not in activated_options:
            for i in indices:
                activated_options[i] = 1.
        activated_options = torch.tensor(activated_options)
        #print(activated_options)
        #print(op_prob)
        #print(op_prob * torch.mul(activated_options,int_func))
        pi_I = op_prob * torch.mul(activated_options,int_func) / torch.sum(op_prob * torch.mul(activated_options,int_func))
        #print(pi_I)
        #option = np.random.choice(range(len(op_prob)), p=pi_I.detach().numpy())
        option = Categorical(pi_I).sample()
        return option

    @property
    def epsilon(self):
        eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
        self.num_steps += 1
        # return eps
        return 0.01


"""

This is the termination function, Beta

"""
class Termination(nn.Module):
    def __init__(self, num_states, num_options):
        super(Termination, self).__init__()
        self.num_states = num_states
        self.num_options = num_options

        self.linear = nn.Linear(self.num_states, self.num_options)
        self.m = nn.Tanh()

    def terminations(self, state):
        termination = self.linear(state)
        # termination = self.m(termination)
        termination = termination.sigmoid()
        return termination

    def predict_option_termination(self, state, current_option):
        termination = self.linear(state)[current_option]
        # termination = self.m(termination)
        termination = termination.sigmoid()
        option_termination = Bernoulli(termination).sample()
        return bool(option_termination.item())


"""

This is a Softmax Policy, one for each option.
This is represented as a 3-D Multilayer Perceptron

"""
class SoftmaxPolicy(nn.Module):
    def __init__(self, num_states, num_options, num_actions, temperature=1):
        super(SoftmaxPolicy, self).__init__()
        self.num_states = num_states
        self.num_options = num_options
        self.num_actions = num_actions
        self.temperature = temperature

        self.options_W = nn.Parameter(torch.zeros(num_options, num_states, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

    def sample_action(self, state, option):
        # logits = state[option] @ self.options_W[option] + self.options_b[option]
        logits = state @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action.item(), logp, entropy


"""

This is the critic model, which is the policy over options and the state model combined.
An individual critic model is not needed and the paarameters of the State Model and
Policy Over Options can be combined, but it kept here in case it is needed to be used.

"""


class CriticModel(nn.Module):
    def __init__(self, policy_over_options, num_states, num_outputs):
        super(CriticModel, self).__init__()
        self.num_states = num_states
        self.num_outputs = num_outputs

        self.features = nn.Sequential(
            nn.Linear(num_states, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.policy_over_options = policy_over_options

    def get_state(self, observation):
        state = self.features(observation)
        return state

    def Q_omega(self, state, option=None):
        # return nn.Sequential(self.linear1, self.act1, self.linear2)
        # state = self.linear1(state)
        # state = self.act1(state)
        # state = self.linear2(state)
        return self.policy_over_options.Q_omega(state)
        # return state

    def sample_option(self, state):
        # if self.rng.uniform() < self.epsilon:
        # return int(self.rng.randint(self.noptions))
        # else:
        # Q = self.get_Q(state)
        # return Q.argmax(dim=-1).item()
        option = self.policy_over_options.linear(state).argmax(dim=-1).item()
        return option


class ReplayBuffer(object):
    def __init__(self, capacity, seed=42):
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, option, reward, next_obs, done):
        self.buffer.append((obs, option, reward, next_obs, done))

    def sample(self, batch_size):
        obs, option, reward, next_obs, done = zip(*self.rng.sample(self.buffer, batch_size))
        return np.stack(obs), option, reward, np.stack(next_obs), done

    def __len__(self):
        return len(self.buffer)


def ActorLoss(obs, next_obs, next_obs_prime, option, logp, entropy, reward, done, option_termination,
              state_model, policy_over_options, policy_over_options_prime):
    # Calculate Actor loss

    state = state_model(obs)
    next_state = state_model(next_obs)
    next_state_prime = state_model(next_obs_prime)

    termination_probs = option_termination.terminations(state)
    option_termination_probs = termination_probs[option]
    next_termination_probs = option_termination.terminations(next_state)
    next_option_termination_prob = next_termination_probs[option]
    # print(next_option_termination_prob)

    gamma = 0.99
    termination_reg = 0.01
    entropy_reg = 0.01

    Q = policy_over_options.Q_omega(state).detach()
    next_Q_prime = policy_over_options_prime.Q_omega(next_state_prime).detach()

    disc_option_termination_probs = next_option_termination_prob.detach()

    gt = reward + (1 - done) * gamma * (1 - disc_option_termination_probs) * next_Q_prime[
        option] + disc_option_termination_probs * next_Q_prime.max(dim=-1)[0]
    gt_det = gt.detach()

    option_Q = Q[option]
    td_errors = gt_det - option_Q
    td_cost = 0.5 * td_errors ** 2

    # Calculate Actor loss. Actor is pi_w + beta + pi_omega
    termination_loss = option_termination_probs * (option_Q.detach() - Q.max(dim=-1)[0].detach() + termination_reg)
    policy_loss = logp * (gt_det - Q.detach()[option].detach()) - entropy_reg * entropy

    actor_loss = termination_loss + policy_loss
    # print(policy_loss)
    # actor_loss.backward()
    # actor_optimizer.step()
    total_loss = actor_loss
    return termination_loss, policy_loss


def CriticLoss(policy_over_options, policy_over_options_prime, option_termination, data_batch):
    obs_critic, options_critic, rewards_critic, next_obs_critic, dones_critic = data_batch
    batch_idx = torch.arange(len(options_critic)).long()
    options_critic = torch.LongTensor(options_critic)
    rewards_critic = torch.FloatTensor(rewards_critic)
    masks_critic = 1 - torch.FloatTensor(dones_critic)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states_critic = torch.tensor(obs_critic)
    Q_critic = policy_over_options.Q_omega(states_critic)

    # the update target contains Q_next, but for stable learning we use prime network for this
    next_states_prime_critic = torch.tensor(next_obs_critic)
    next_Q_prime_critic = policy_over_options_prime.Q_omega(next_states_prime_critic)  # detach?

    # Additionally, we need the beta probabilities of the next state
    next_states_critic = torch.tensor(next_obs_critic)
    next_termination_probs_critic = option_termination.terminations(next_states_critic)
    next_options_term_prob_critic = next_termination_probs_critic[batch_idx, options_critic]

    # Now we can calculate the update target gt
    gamma = 0.99
    gt_critic = rewards_critic + masks_critic * gamma * \
                ((1 - next_options_term_prob_critic) * next_Q_prime_critic[
                    batch_idx, options_critic] + next_options_term_prob_critic * next_Q_prime_critic.max(dim=-1)[0])

    # print(gt)
    # to update Q we want to use the actual network, not the prime
    td_err_critic = (Q_critic[batch_idx, options_critic] - gt_critic.detach()).pow(2).mul(0.5).mean()
    return td_err_critic


def criticloss(policy_over_options, policy_over_options_prime, option_termination, data_batch):
    state, options, rewards, next_state, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options = torch.LongTensor(options)
    rewards = torch.FloatTensor(rewards)
    masks = 1 - torch.FloatTensor(dones)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    # states = model.get_state(to_tensor(obs)).squeeze(0)
    states = torch.tensor(state)
    Q = policy_over_options.Q_omega(states)

    # the update target contains Q_next, but for stable learning we use prime network for this
    next_states_prime = torch.tensor(next_state)
    next_Q_prime = policy_over_options.Q_omega(next_states_prime)  # detach?

    # Additionally, we need the beta probabilities of the next state
    next_states = torch.tensor(next_state)
    next_termination_probs = option_termination.terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]
    gamma = 0.99
    # Now we can calculate the update target gt
    gt = rewards + masks * gamma * \
         ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob *
          next_Q_prime.max(dim=-1)[0])

    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err


def actorloss(state, option, logp, entropy, reward, done, next_state, option_termination,
              state_model, policy_over_options, policy_over_options_prime):
    next_state_prime = next_state

    option_term_prob = option_termination.terminations(state)[option]
    next_option_term_prob = option_termination.terminations(next_state)[option].detach()

    Q = policy_over_options.Q_omega(state).detach().squeeze()
    next_Q_prime = policy_over_options.Q_omega(next_state_prime).detach().squeeze()
    gamma = 0.99
    termination_reg = 0.01
    entropy_reg = 0.01

    # Target update gt
    gt = reward + (1 - done) * gamma * \
         ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob * next_Q_prime.max(dim=-1)[0])

    # The termination loss
    gt_detached = gt.detach()
    Q_detached = Q.detach()
    # termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + termination_reg) * (1 - done)

    termination_loss = (option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + termination_reg)).sum()

    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp * (gt_detached - Q[option]) - entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    # return actor_loss
    return termination_loss, policy_loss


# Notes:
"""
1) Add softmax to policy_over_options
2) Consider adding tanh to the other OCs
3) Consider adding [] when passing option to linear, maybe to accomodate higher batches
"""