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

import random

import interface

# Choose cpu or gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Actions for moving
ACTIONS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1), "attack", "connect")


# Define the neural network for the policy
class Policy(nn.Module):
    def __init__(self, s_size, a_size, layers_data: list):
        super(Policy, self).__init__()

        self.layers = nn.ModuleList()
        input_size = s_size
        for size, activation in layers_data[0]:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size
            if activation is not None:
                assert isinstance(activation, Module), \
                    "Each tuple should contain a layer size (int) and an activation (ex. nn.ReLU())."
                self.layers.append(activation)
        self.layers.append(nn.Linear(size, a_size[0]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.softmax(x, dim=-1)
    
    def act(self, state, map, cx, cy, lighthouses):
        """
        Given a state, take action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        valid_moves = [(x,y) for x,y in ACTIONS[:8] if map[cy+y][cx+x]]
        valid_indices = [ACTIONS.index(i) for i in ACTIONS if i in valid_moves]
        if (cx, cy) in lighthouses:
            valid_indices = valid_indices + [8,9]
        indices = torch.tensor(valid_indices)
        probs = probs[0]
        probs_valid = probs[indices]
        is_all_zeros = torch.all(probs_valid == 0)
        if is_all_zeros.item():
            probs_valid = probs_valid + 0.0000001
        m = Categorical(probs_valid)
        action = m.sample()
        return valid_indices[action.item()], m.log_prob(action)

    
    # Create training loop
class REINFORCE(interface.Bot):
    def __init__(self):
        self.NAME = "REINFORCE",
        self.a_size = len(ACTIONS),
        self.layers_data = [(16, nn.ReLU())],
        self.n_training_episodes = 100,
        self.n_evaluation_episodes = 10,
        self.max_t = 1000,
        self.gamma = 1.0,
        self.lr = 1e-2,
        self.save_model = True, 
        self.model_path = './saved_model',
        self.print_every = 100,
        self.s_size = None,
        self.policy = None,
        self.optimizer = None,
        self.device = device
        self.use_saved_model = False
    
    def initialize_game(self, state):
        state = self.convert_state(state)
        self.s_size = len(state)
        self.policy = Policy(self.s_size, self.a_size, self.layers_data).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr[0])
        if self.use_saved_model:
            self.load_saved_model()


    def convert_state(self, state):
        cx = state['position'][0]
        cy = state['position'][1]
        lighthouses =[]
        for lh in state['lighthouses']:
            lighthouses.append(lh['position'][0])
            lighthouses.append(lh['position'][1])
            lighthouses.append(lh['energy'])
        new_state = np.array([state['position'][0], state['position'][1], state['energy'], len(state['lighthouses'])] + lighthouses)

        return new_state


    def play(self, state):
        #lighthouses = dict((tuple(lh["position"]), lh) for lh in state["lighthouses"])
        cx = state['position'][0]
        cy = state['position'][1]
        lighthouses = [(tuple(lh['position'])) for lh in state["lighthouses"]]
        new_state = self.convert_state(state)
        action, _ = self.policy.act(new_state, self.map, cx, cy, lighthouses)
        if ACTIONS[action] != "attack" and ACTIONS[action] != "connect":
            return self.move(*ACTIONS[action])
        elif ACTIONS[action] == "attack":
            energy = random.randrange(state["energy"] + 1)
            return self.attack(energy)
        elif ACTIONS[action] == "connect":
            return self.connect(random.choice(lighthouses))
    

    def play_train(self, state):
         #lighthouses = dict((tuple(lh["position"]), lh) for lh in state["lighthouses"])
        cx = state['position'][0]
        cy = state['position'][1]
        lighthouses = [(tuple(lh['position'])) for lh in state["lighthouses"]]
        new_state = self.convert_state(state)
        action, log_prob = self.policy.act(new_state, self.map, cx, cy, lighthouses)
        if ACTIONS[action] != "attack" and ACTIONS[action] != "connect":
            return self.move(*ACTIONS[action]), log_prob
        elif ACTIONS[action] == "attack":
            energy = random.randrange(state["energy"] + 1)
            return self.attack(energy), log_prob
        elif ACTIONS[action] == "connect":
            return self.connect(random.choice(lighthouses)), log_prob
    

    def load_saved_model(self):
        if os.path.exists(os.path.dirname(self.model_path)):
                if os.path.isfile(self.model_path+'/reinforce.pth'):
                    self.policy.load_state_dict(torch.load(self.model_path+'/reinforce.pth'))
                    print("Loaded saved model")
        else:
            print("No saved model")


    def calculate_returns(self, rewards, max_t):
        # Calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards) 
                
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(self.gamma[0]*disc_return_t + rewards[t])
        # Standardize returns to make training more stable
        eps = np.finfo(np.float32).eps.item()

        # eps is the smallest representable float which is added to the standard
        # deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean())/(returns.std() + eps)

        return returns


    def update_policy(self, rewards, max_t, saved_log_probs):
        # Calculate the loss and update the policy weights

        returns = self.calculate_returns(rewards, max_t)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Calculate the gradient and update the weights
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(self.policy.state_dict(), self.model_path+'/reinforce.pth')
        print("Saved model to disk")
