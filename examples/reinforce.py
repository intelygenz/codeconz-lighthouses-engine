import numpy as np
import os

from collections import deque

import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules import Module
from torch.distributions import Categorical

# Gym
import gym
import gym_pygame

# Replay video
import imageio


# Choose cpu or gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env_id = "CartPole-v1"
# Create the env
env = gym.make(env_id)

# Create the evaluation env
eval_env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

# Define the neural network for the policy
class Policy(nn.Module):
    def __init__(self, s_size, a_size, layers_data: list):
        super(Policy, self).__init__()

        self.layers = nn.ModuleList()
        input_size = s_size
        for size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size
            if activation is not None:
                assert isinstance(activation, Module), \
                    "Each tuple should contain a layer size (int) and an activation (ex. nn.ReLU())."
                self.layers.append(activation)
        self.layers.append(nn.Linear(size, a_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.softmax(x, dim=-1)
    
    def act(self, state):
        """
        Given a state, take action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        ### ADD IN REQUISITE THAT THE ACTION IS POSSIBLE
        return action.item(), m.log_prob(action)
    
    # Create training loop
class REINFORCE:
    def __init__(self, env, policy, optimizer, gamma, model_path='./saved_model'):
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.model_path = model_path


    def load_saved_model(self):
        if os.path.exists(os.path.dirname(self.model_path)):
                if os.path.isfile(self.model_path+'/reinforce.pth'):
                    self.policy.load_state_dict(torch.load(self.model_path+'/reinforce.pth'))
                    print("Loaded saved model")
        else:
            print("No saved model")


    def train(self, n_training_episodes, max_t, print_every=100, save_model=True, use_saved_model=False):
        # Help calculate score during training
        # Line 3 of pseudocode
        self.max_t = max_t
        scores_deque = deque(maxlen=100)
        scores = []

        if use_saved_model:
            self.load_saved_model()

        # Repeat: Generate an episode, calculte the return based on the steps that remain, calculate loss, update gradient using loss
        for i_episode in range(1, n_training_episodes+1):
            self.saved_log_probs = []
            self.rewards = []
            state = self.env.reset()[0]
            # Generate an episode following the policy
            for t in range(max_t):
                action, log_prob = self.policy.act(state)
                self.saved_log_probs.append(log_prob)
                state, reward, done, _, _ = self.env.step(action)
                self.rewards.append(reward)
                if done:
                    break
            scores_deque.append(sum(self.rewards))
            scores.append(sum(self.rewards))

            # Calculate the returns
            returns = self.calculate_returns()

            # Update the policy
            self.update_policy(returns)

            if i_episode % print_every == 0:
                print('Episode{}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
        if save_model:
            os.makedirs(self.model_path, exist_ok=True)
            torch.save(self.policy.state_dict(), self.model_path+'/reinforce.pth')
            print("Saved model to disk")

        return scores


    def calculate_returns(self):
        # Calculate the return
        returns = deque(maxlen=self.max_t)
        n_steps = len(self.rewards) 
                
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(self.gamma*disc_return_t + self.rewards[t])
        # Standardize returns to make training more stable
        eps = np.finfo(np.float32).eps.item()

        # eps is the smallest representable float which is added to the standard
        # deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean())/(returns.std() + eps)

        return returns


    def update_policy(self, returns):
        # Calculate the loss and update the policy weights
        policy_loss = []
        for log_prob, disc_return in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Calculate the gradient and update the weights
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        
    def play(self, state, max_steps, n_eval_episodes, use_saved_model=True):
        """
        Play for ``n_eval_episodes`` episodes and returns average reward and std of reward.
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param max_steps: Max number of steps in episode
        """ 
        if use_saved_model:
            self.load_saved_model()
            
        self.policy.eval()  
        episode_rewards = []
        for episode in range(n_eval_episodes):
            state = self.env.reset()[0]
            step = 0
            done = False
            total_rewards_ep = 0

            for step in range(max_steps):
                action, _ = self.policy.act(state)
                new_state, reward, done, info, _ = self.env.step(action)
                total_rewards_ep += reward

                if done:
                    break
                state = new_state
            episode_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)  
        print(f'Average reward: {mean_reward}, Standard deviation: {std_reward}')  

        return mean_reward, std_reward

cartpole_hyperparameters = {
    "state_space": s_size,
    "action_space": a_size,
    "layers_data": [(16, nn.ReLU())],
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "state_space": s_size,
    "action_space": a_size,
    "save_model": True, 
    "model_path": './saved_model',
    "print_every": 100,
}

cartpole_policy = Policy(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["layers_data"]).to(device)
cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

reinforce = REINFORCE(env, cartpole_policy, cartpole_optimizer, cartpole_hyperparameters["gamma"])

scores = reinforce.train(cartpole_hyperparameters["n_training_episodes"], cartpole_hyperparameters["max_t"], cartpole_hyperparameters["print_every"])

avg, stddev = reinforce.play(cartpole_hyperparameters["max_t"], cartpole_hyperparameters["n_evaluation_episodes"])