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
import os
from models import *
from fourrooms import *
from twod_tmaze import TMaze

def train():
    env_name = 'FourroomsOnehot-v0'
    #env = TMaze()
    #env.seed(2)
    env = gym.make(env_name)
    print(env)
    num_actions  = env.action_space.n
    num_states = env.observation_space.n
    print(num_states)
    num_options = 4
    actionSpace = range(num_actions)

    # Define everything that needs to be learned, and their hyperparameters
    lr_critic = 7e-4 #1e-5
    lr_intra = 7e-3 #7e-4
    lr_termination = 7e-5 #1e-4
    gamma = 0.99
    termination_reg = 0.01
    entropy_reg = 0.01

    momentum = 0.9
    alpha = 0.95
    obs = env.reset()
    print(len(obs))
    state_model = StateModel(len(obs), 32)
    #policy_over_options = PolicyOverOptions(123, len(obs), num_options, eps_decay=10000)
    policy_over_options = FixedPolicyOverOptions(123, len(obs), num_options, eps_decay=10000)
    interest_policy_over_options = InterestPolicyOverOptions(123, len(obs), num_options, policy_over_options, eps_decay=10000)
    interest_policy_over_options_prime = InterestPolicyOverOptions(123, len(obs), num_options, policy_over_options)
    option_termination = Termination(len(obs), num_options)
    option_policies = SoftmaxPolicy(len(obs), num_options, num_actions)
    #critic = CriticModel(policy_over_options, num_states, 64)
    max_episodes = 5000000
    max_steps_per_episode = 50

    critic_params = list(interest_policy_over_options.parameters()) # + list(state_model.parameters())
    critic_optimizer = optim.RMSprop(critic_params, lr=lr_critic, alpha=alpha)
    termination_optimizer = optim.RMSprop(option_termination.parameters(), lr=lr_termination)
    intra_optimizer = optim.RMSprop(option_policies.parameters(), lr=lr_intra, alpha=alpha)
    actor_params = list(option_termination.parameters()) + list(option_policies.parameters())
    actor_optimizer = optim.RMSprop(actor_params, lr=lr_termination)

    #critic_params = list(interest_policy_over_options.parameters())  # + list(state_model.parameters())
    #critic_optimizer = optim.Adam(critic_params, lr=lr_critic)
    #termination_optimizer = optim.Adam(option_termination.parameters(), lr=lr_termination)
    #intra_optimizer = optim.Adam(option_policies.parameters(), lr=lr_intra)
    #actor_params = list(option_termination.parameters()) + list(option_policies.parameters())
    #actor_optimizer = optim.Adam(actor_params, lr=lr_termination)
  
    # To be used when combining all 3 elements
    #total_params = list(interest_policy_over_options.parameters()) + list(state_model.parameters()) + list(option_termination.parameters()) + list(option_policies.parameters())
    total_params = list(interest_policy_over_options.parameters()) + list(option_termination.parameters()) + list(option_policies.parameters())
    total_optimizer = optim.RMSprop(total_params, lr=lr_critic)

    buffer = ReplayBuffer(capacity=10000, seed=0)

    episode_performances = []
    episode_switches = []
    episode_start = []
    episode_steps_arr = []
    num_steps = 0
    print("USING THE SPLIT MODELS")
    obs = env.reset()
    state = torch.tensor(obs).float()
    print(interest_policy_over_options.sample_option(state))
    for episode in range(max_episodes):
        obs = env.reset()
        episode_start.append(obs)

        # If using (i, j, distance)
        state = torch.tensor(obs).float()
        #state = state_model(torch.tensor(obs).float())

        rewards = 0
        #print("One episode")
        done = False
        num_switches = 0
        episode_steps = 0
        termination = True
        option_steps = 0
        option = interest_policy_over_options.sample_option(state)
        while done == False:
            #print("One step")
            num_steps += 1
            episode_steps += 1

            if (termination):
                option = interest_policy_over_options.sample_option(state)
                num_switches = num_switches + 1
                option_steps = 0
      
            action, logp, entropy = option_policies.sample_action(state, option)
            option_steps += 1
            next_obs, reward, done, _ = env.step(action)
            rewards = reward + rewards
            #next_obs = torch.from_numpy(next_obs).float()
            #next_state = critic.get_state(next_obs)
            #next_state = state_model(next_obs)

            # If using (i, j, distance)
            next_state = torch.tensor(next_obs).float()
            #next_state = state_model(torch.tensor(next_obs).float())

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
                termination_loss, policy_loss = actorloss(state, option, logp, entropy, reward, done, next_state, option_termination, state_model, interest_policy_over_options, interest_policy_over_options)
                #actor_loss = termination_loss + policy_loss
                #actor_loss.backward()
                #actor_optimizer.step()
                #print("termination loss = ", termination_loss.item(), " policy loss = ", policy_loss.item())

                #### RUN ACTOR_CRITIC BY SPLITTING INTO 3 PARTS
                termination_optimizer.zero_grad()
                intra_optimizer.zero_grad()
                termination_loss, policy_loss = actorloss(state, option, logp, entropy, reward, done, next_state, option_termination, state_model, interest_policy_over_options, interest_policy_over_options_prime)
                actor_loss = termination_loss + policy_loss
                termination_loss.backward()
                termination_optimizer.step()
                #intra_optimizer.zero_grad()
                policy_loss.backward()
                intra_optimizer.step()


                #### RUN ACTOR CRITIC FOR TOTAL LOSS
                total_loss = termination_loss + policy_loss
                #total_loss = actor_loss

                # Calculate Critic Loss. Critic is Q_u + State Model. The Critic can be updated less frequently
                if (num_steps % 10 == 0):
                    data_batch = buffer.sample(32)

                    #critic_cost = CriticLoss(policy_over_options, option_termination, data_batch)
                    critic_optimizer.zero_grad()
                    critic_cost = criticloss(interest_policy_over_options, interest_policy_over_options_prime, option_termination, data_batch)
                    critic_cost.backward()
                    critic_optimizer.step()

                    #total_loss = total_loss + critic_cost

                #total_optimizer.zero_grad()
                #total_loss.backward()
                #total_optimizer.step()
    
                if num_steps % 200 == 0:
                    interest_policy_over_options.load_state_dict(interest_policy_over_options.state_dict())
      
            state = next_state
            termination = option_termination.predict_option_termination(state, option)


        episode_performances.append(rewards)
        episode_switches.append(num_switches)
        episode_steps_arr.append(episode_steps)

        if len(episode_performances) >= 100:
            avg_rewards = np.mean(episode_performances[-50:])
            avg_switches = np.mean(episode_switches[-50:])
            avg_steps = np.mean(episode_steps_arr[-50:])
        else:
            avg_rewards = 0
            avg_switches = num_switches
            avg_steps = 1000

        #if episode == 200:
        #    env.reset_goal()
        #    print("New goal is = ", env.goal)


        if episode % 50 == 0:
            epsilon = interest_policy_over_options.epsilon
            #print(episode_performances[-50:])
            #print(episode_start[-50:])
            print("Steps = ", num_steps, "Epsilon = ", epsilon, " Episode Steps = ", episode_steps)
            if (critic_cost == None):
                print(f"[{episode} (0)]\tActor_Loss: {actor_loss.item():.4f}\tSwitches: {avg_switches}\t\tAvg: {avg_rewards}")
            else:
                print(f"[{episode} (0)]\tCritic_Loss: {critic_cost.item():.4f}\tActor_Loss: {actor_loss.item():.4f}\tSwitches: {avg_switches}\t\tAvg: {avg_rewards}")

        # Checks if solved
        file_num = 11
        rooms = env_name.split("-")[0] # "TwoRooms"
        path = "Trained_Models/" + rooms + "/" + str(file_num) + "/"
        if not os.path.exists(path):
            os.makedirs(path)

        if len(episode_steps_arr) == 500:
            #if avg_steps <= 50:
            #if avg_rewards >= 0.8:
            print("Done Training")
            print(avg_rewards)
            print(avg_switches)
            print(avg_steps)
            print("Steps = ", episode_steps_arr)
            print("Switches = ", episode_switches)
            torch.save(state_model.state_dict(), path + "State_model.pt")
            torch.save(policy_over_options.state_dict(),path + "Policy_over_options.pt")
            torch.save(interest_policy_over_options.state_dict(), path + "Interest_Policy_over_options.pt")
            torch.save(option_termination.state_dict(), path + "Option_termination.pt")
            torch.save(option_policies.state_dict(), path + "Option_policies.pt")
            torch.save(torch.tensor(episode_steps_arr), path + "episode_steps.pt")
            return episode

        if episode % 50 == 0 and episode != 0:
            print("Saving Model")
            print(avg_rewards)
            print(avg_switches)
            print(avg_steps)
            torch.save(state_model.state_dict(), path + "State_model.pt")
            torch.save(policy_over_options.state_dict(),path + "Policy_over_options.pt")
            torch.save(interest_policy_over_options.state_dict(), path + "Interest_Policy_over_options.pt")
            torch.save(option_termination.state_dict(), path + "Option_termination.pt")
            torch.save(option_policies.state_dict(), path + "Option_policies.pt")

train()
