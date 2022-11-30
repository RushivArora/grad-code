import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.distributions import Categorical, Bernoulli

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from math import exp

from fourrooms import Fourrooms

"""

The state model takes in a observation, and converts it into a 64 dimension tensor

"""
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
        nn.Linear(num_states, 32),
        nn.ReLU()
    )


  def forward(self, observation):
    state = self.features(observation)
    return state.detach()

"""

This is the policy over options, the Q function

"""
class PolicyOverOptions(nn.Module):
  def __init__(self, rng, num_states, num_options, eps_start=1.0, eps_min=0.1, eps_decay=int(1e5)):
    super(PolicyOverOptions, self).__init__()
    self.rng = rng
    self.num_states = num_states
    self.num_options = num_options
    #self.epsilon = epsilon

    #self.linear1 = nn.Linear(self.num_states, 64)
    #self.act1 = nn.ReLU()
    #self.linear2 = nn.Linear(64, self.num_states)

    self.eps_min   = eps_min
    self.eps_start = eps_start
    self.eps_decay = eps_decay
    self.num_steps = 0

    self.linear = nn.Linear(self.num_states, self.num_options)

  def Q_omega(self, state, option=None):
    #return nn.Sequential(self.linear1, self.act1, self.linear2)
    #state = self.linear1(state)
    #state = self.act1(state)
    #state = self.linear2(state)
    return self.linear(state)
    #return state

  def sample_option(self, state):
    #if self.rng.uniform() < self.epsilon:
      #return int(self.rng.randint(self.noptions))
    #else:
    #Q = self.get_Q(state)
    #return Q.argmax(dim=-1).item()
    epsilon = self.epsilon
    e_greedy_option = self.linear(state).argmax(dim=-1).item()
    option = np.random.choice(self.num_options) if np.random.rand() < epsilon else e_greedy_option
    return option
  
  @property
  def epsilon(self):
    eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
    self.num_steps += 1
    return eps
    #return 0.1


"""

This is the termination function, Beta

"""
class Termination(nn.Module):
  def __init__(self, num_states, num_options):
    super(Termination, self).__init__()
    self.num_states = num_states
    self.num_options = num_options
    
    #self.linear1 = nn.Linear(self.num_states, 64)
    #self.act1 = nn.ReLU()
    #self.linear2 = nn.Linear(64, self.num_states)
    #self.sig = nn.Sigmoid()
        
    self.linear = nn.Linear(self.num_states, self.num_options)

  def terminations(self, state):
    #state = self.linear1(state)
    #state = self.act1(state)
    #state = self.linear2(state).sigmoid()
    #state = self.sig(state)
    state = self.linear(state).sigmoid()
    return state

  def predict_option_termination(self, state, current_option):
    termination = self.linear(state)[current_option].sigmoid()
    #state = self.linear1(state)
    #state = self.act1(state)
    #termination = self.linear2(state).sigmoid()
    option_termination = Bernoulli(termination).sample()
    return bool(option_termination.item())


"""

This is a Ssoftmax Policy, one for each option.
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
    #logits = state[option] @ self.options_W[option] + self.options_b[option]
    logits = state @ self.options_W[option] + self.options_b[option]
    action_dist = (logits / self.temperature).softmax(dim=-1)
    action_dist = Categorical(action_dist)

    action  = action_dist.sample()
    logp    = action_dist.log_prob(action)
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
        nn.Linear(32,64),
        nn.ReLU()
    )

    self.policy_over_options = policy_over_options

  def get_state(self, observation):
    state = self.features(observation)
    return state
  
  def Q_omega(self, state, option=None):
    #return nn.Sequential(self.linear1, self.act1, self.linear2)
    #state = self.linear1(state)
    #state = self.act1(state)
    #state = self.linear2(state)
    return self.policy_over_options.Q_omega(state)
    #return state

  def sample_option(self, state):
    #if self.rng.uniform() < self.epsilon:
      #return int(self.rng.randint(self.noptions))
    #else:
    #Q = self.get_Q(state)
    #return Q.argmax(dim=-1).item()
    option = self.policy_over_options.linear(state).argmax(dim=-1).item()
    return option

import numpy as np
import random
from collections import deque

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
  #print(next_option_termination_prob)

  gamma = 0.99
  termination_reg = 0.01
  entropy_reg = 0.01

  Q = policy_over_options.Q_omega(state).detach()
  next_Q_prime = policy_over_options_prime.Q_omega(next_state_prime).detach()

  disc_option_termination_probs = next_option_termination_prob.detach()

  gt = reward + (1-done)*gamma*(1 - disc_option_termination_probs)*next_Q_prime[option] + disc_option_termination_probs*next_Q_prime.max(dim=-1)[0]
  gt_det = gt.detach()

  option_Q = Q[option]
  td_errors = gt_det - option_Q
  td_cost = 0.5 * td_errors ** 2

  # Calculate Actor loss. Actor is pi_w + beta + pi_omega
  termination_loss = option_termination_probs*(option_Q.detach() - Q.max(dim=-1)[0].detach() + termination_reg)
  policy_loss = logp*(gt_det - Q.detach()[option].detach()) - entropy_reg*entropy

  actor_loss = termination_loss + policy_loss
  #print(policy_loss)
  #actor_loss.backward()
  #actor_optimizer.step()
  total_loss = actor_loss
  return termination_loss, policy_loss

def CriticLoss(policy_over_options, option_termination, data_batch):
  obs_critic, options_critic, rewards_critic, next_obs_critic, dones_critic = data_batch
  batch_idx = torch.arange(len(options_critic)).long()
  options_critic  = torch.LongTensor(options_critic)
  rewards_critic   = torch.FloatTensor(rewards_critic)
  masks_critic     = 1 - torch.FloatTensor(dones_critic)
  
  # The loss is the TD loss of Q and the update target, so we need to calculate Q
  states_critic = torch.tensor(obs_critic)
  Q_critic      = policy_over_options.Q_omega(states_critic)
  
  # the update target contains Q_next, but for stable learning we use prime network for this
  next_states_prime_critic = torch.tensor(next_obs_critic)
  next_Q_prime_critic      = policy_over_options.Q_omega(next_states_prime_critic) # detach?

  # Additionally, we need the beta probabilities of the next state
  next_states_critic           = torch.tensor(next_obs_critic)
  next_termination_probs_critic = option_termination.terminations(next_states_critic)
  next_options_term_prob_critic = next_termination_probs_critic [batch_idx, options_critic]
  
  # Now we can calculate the update target gt
  gamma = 0.99
  gt_critic = rewards_critic + masks_critic * gamma * \
      ((1 - next_options_term_prob_critic) * next_Q_prime_critic[batch_idx, options_critic] + next_options_term_prob_critic  * next_Q_prime_critic.max(dim=-1)[0])
      
  #print(gt)
  # to update Q we want to use the actual network, not the prime
  td_err_critic = (Q_critic[batch_idx, options_critic] - gt_critic.detach()).pow(2).mul(0.5).mean()
  return td_err_critic
  
def criticloss(policy_over_options, option_termination, data_batch):
    state, options, rewards, next_state, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options   = torch.LongTensor(options)
    rewards   = torch.FloatTensor(rewards)
    masks     = 1 - torch.FloatTensor(dones)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    #states = model.get_state(to_tensor(obs)).squeeze(0)
    states = torch.tensor(state)
    Q      = policy_over_options.Q_omega(states)
    
    # the update target contains Q_next, but for stable learning we use prime network for this
    next_states_prime = torch.tensor(next_state)
    next_Q_prime      = policy_over_options.Q_omega(next_states_prime) # detach?

    # Additionally, we need the beta probabilities of the next state
    next_states            = torch.tensor(next_state)
    next_termination_probs = option_termination.terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]
    gamma = 0.99
    # Now we can calculate the update target gt
    gt = rewards + masks * gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob * next_Q_prime.max(dim=-1)[0])

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
        ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob  * next_Q_prime.max(dim=-1)[0])

    # The termination loss
    gt_detached = gt.detach()
    Q_detached = Q.detach()
    termination_loss = option_term_prob * (Q[option].detach() - Q.max(dim=-1)[0].detach() + termination_reg) * (1 - done)
    
    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp * (gt_detached - Q[option]) - entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    #return actor_loss
    return termination_loss, policy_loss

class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

def train():
  env_name = 'CartPole-v0'
  env = gym.make(env_name)
  env = Fourrooms()
  print(env)
  num_actions  = env.action_space.n
  #num_actions  = 3
  num_states = env.observation_space.shape[0]
  #num_states = env.observation_space.n
  #features = Tabular(env.observation_space.n)
  print(num_states)
  num_options = 2
  actionSpace = range(num_actions)
  optionSpace = range(num_options)

  # Define everything that needs to be learned, and their hyperparameters
  lr_critic = 0.05
  lr_intra = 0.0025
  lr_termination = 0.0025
  gamma = 1
  termination_reg = 0.01
  entropy_reg = 0.01

  momentum = 0.9
  alpha = 0.95
  state_model = StateModel(num_states, 32)
  policy_over_options = PolicyOverOptions(123, 32, num_options, eps_decay=20000)
  policy_over_options_prime = PolicyOverOptions(123, 32, num_options)
  option_termination = Termination(32, num_options)
  option_policies = SoftmaxPolicy(32, num_options, num_actions)
  #critic = CriticModel(policy_over_options, num_states, 64)
  max_episodes = 100000
  max_steps_per_episode = 1000

  critic_params = list(policy_over_options.parameters()) + list(state_model.parameters())
  critic_optimizer = optim.RMSprop(critic_params, lr=lr_critic)
  termination_optimizer = optim.RMSprop(option_termination.parameters(), lr=lr_termination)
  intra_optimizer = optim.RMSprop(option_policies.parameters(), lr=lr_intra)
  actor_params = list(option_termination.parameters()) + list(option_policies.parameters())
  actor_optimizer = optim.RMSprop(actor_params, lr=lr_termination)
  
  total_params = list(policy_over_options.parameters()) + list(state_model.parameters()) + list(option_termination.parameters()) + list(option_policies.parameters())
  total_optimizer = optim.RMSprop(total_params, lr=lr_critic)
  """
  total_optimizer = optim.RMSprop([
                                    {'params': policy_over_options.parameters()},
                                    {'params': state_model.parameters()},
                                    {'params': option_termination.parameters(), 'lr': lr_termination},
                                    {'params': option_policies.parameters(), 'lr': lr_intra}
                                ], lr=lr_critic)
    """

  buffer = ReplayBuffer(capacity=10000, seed=0)

  episode_performances = []
  episode_switches = []
  num_steps = 0
  print("USING THE SPLIT MODELS")
  for episode in range(max_episodes):
    obs = env.reset()
    #phi = features(obs)
    #print(obs)
    #print("Phi = ", phi)
    obs = torch.from_numpy(obs).float()
    #state = critic.get_state(obs)
    #s = np.zeros(env.observation_space.n)
    #s[obs] = 1
    #s = torch.from_numpy(s).float()
    #state = state_model(s)
    state = state_model(obs)
    #print(state)

    rewards = 0
    #print("One episode")
    done = False
    num_switches = 0
    termination = True
    while done == False:
      #print("One step")
      num_steps += 1
      
      #Calculate Critic loss
      #critic_optimizer.zero_grad()
      #termination_optimizer.zero_grad()
      #intra_optimizer.zero_grad()
      #actor_optimizer.zero_grad()
      #total_optimizer.zero_grad()

      if (termination):
        option = policy_over_options.sample_option(state)
        num_switches = num_switches + 1
      
      action, logp, entropy = option_policies.sample_action(state, option)

      next_obs, reward, done, _ = env.step(action)
      rewards = reward + rewards
      next_obs = torch.from_numpy(next_obs).float()
      #next_state = critic.get_state(next_obs)
      next_state = state_model(next_obs)
      #next_s = np.zeros(env.observation_space.n)
      #next_s[next_obs] = 1
      #next_s = torch.from_numpy(next_s).float()
      #next_state = state_model(next_s)

      buffer.push(state, option, reward, next_state, done)

      critic_cost = None
      actor_loss = torch.tensor(0)
      if (num_steps > 32):

        # actor-critic policy gradient with entropy regularization
        #termination_loss.backward()
        #termination_optimizer.step()
        #policy_loss.backward()
        #intra_optimizer.step()
        #termination_loss, policy_loss = ActorLoss(obs, next_obs, next_obs, option, logp, entropy, reward, done, option_termination,  state_model, policy_over_options, policy_over_options)
        #actor_loss = termination_loss + policy_loss
        
        #actor_optimizer.zero_grad()
        #actor_loss = actorloss(state, option, logp, entropy, reward, done, next_state, option_termination, state_model, policy_over_options, policy_over_options)
        #actor_loss.backward()
        #actor_optimizer.step()
        #print("termination loss = ", termination_loss.item(), " policy loss = ", policy_loss.item())
        
        termination_optimizer.zero_grad()
        termination_loss, policy_loss = actorloss(state, option, logp, entropy, reward, done, next_state, option_termination, state_model, policy_over_options, policy_over_options)
        termination_loss.backward()
        termination_optimizer.step()
        intra_optimizer.zero_grad()
        policy_loss.backward()
        intra_optimizer.step()
        #total_loss = termination_loss + policy_loss
        #total_loss = actor_loss

        # Calculate Critic Loss. Critic is Q_u + State Model. The Critic can be updated less frequently
        if (num_steps % 4 == 0):
          data_batch = buffer.sample(32)

          #critic_cost = CriticLoss(policy_over_options, option_termination, data_batch)
          critic_optimizer.zero_grad()
          critic_cost = criticloss(policy_over_options, option_termination, data_batch)
          critic_cost.backward()
          critic_optimizer.step()
          
          #total_loss = total_loss + critic_cost
    
        #total_loss.backward()
        #total_optimizer.step()
      
      state = next_state
      termination = option_termination.predict_option_termination(state, option)


    episode_performances.append(rewards)
    episode_switches.append(num_switches)
    if len(episode_performances) >= 50:
      avg_rewards = np.mean(episode_performances[-50:])
      avg_switches = np.mean(episode_switches[-50:])
    else:
      avg_rewards = 0
      avg_switches = num_switches

    if episode % 50 == 0:
      epsilon = policy_over_options.epsilon
      print("Steps = ", num_steps, "Epsilon = ", epsilon)
      if (critic_cost == None):
        #print(f"[{episode} (0)]\tpi_loss: {critic_cost.item():.4f}\t\tv_loss: {termination_loss.item():.4f}\t\tv_loss: {policy_loss.item():.4f}\t\tAvg: {avg_rewards}")
        print(f"[{episode} (0)]\tActor_Loss: {actor_loss.item():.4f}\tSwitches: {avg_switches}\t\tAvg: {avg_rewards}")
        #print("Okay")
      else:
        print(f"[{episode} (0)]\tCritic_Loss: {critic_cost.item():.4f}\tActor_Loss: {actor_loss.item():.4f}\tSwitches: {avg_switches}\t\tAvg: {avg_rewards}")
    # Checks if solved
    if avg_rewards >= 190:
      print("Done Training")
      print(avg_rewards)
      print(avg_switches)
      return episode

train()
