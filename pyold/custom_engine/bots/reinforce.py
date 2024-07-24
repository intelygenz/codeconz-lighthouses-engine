#!/usr/bin/python
# -*- coding: utf-8 -*-

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
from sklearn.preprocessing import StandardScaler

import random

from bots import bot

# Choose cpu or gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Actions for moving
ACTIONS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1), "attack", "connect", "pass")

# Define the neural network for the policy
class PolicyMLP(nn.Module):
    def __init__(self, s_size, a_size, layers_data: list):
        super(PolicyMLP, self).__init__()

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
        return torch.log_softmax(x, dim=-1)
    
    def act(self, state):
        """
        Given a state, take action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
class PolicyCNN(nn.Module):
    def __init__(self, num_maps, a_size: list):
        super(PolicyCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=num_maps, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*13*13, 32),
            nn.ReLU(),
            nn.Linear(32, a_size)
        )

    def forward(self, x):
        x = self.network(x)
        return torch.log_softmax(x, dim=-1)
    
    def act(self, state):
        """
        Given a state, take action
        """
        state = np.transpose(state, (2,0,1))
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    
# Create training loop
class REINFORCE(bot.Bot):
    def __init__(self, state_maps=True, model_filename='model.pth', use_saved_model=True):
        super().__init__()
        self.NAME = "REINFORCE"
        self.state_maps = state_maps # use maps for state: True, or array for state: False
        self.a_size = len(ACTIONS)
        self.layers_data = [(16, nn.ReLU())]
        self.gamma = 1.0
        self.lr = 1e-2
        self.save_model = True 
        self.model_path = './saved_model'
        self.model_filename = model_filename
        self.use_saved_model = use_saved_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def initialize_game(self, state):
        self.saved_log_probs = []
        if self.state_maps:
            print("Using maps for state: PolicyCNN")
            state = self.convert_state_cnn(state)
            self.num_maps = state.shape[2]
            self.policy = PolicyCNN(self.num_maps, self.a_size).to(self.device)
        else:
            print("Using array for state: PolicyMLP")
            state = self.convert_state_mlp(state)
            self.s_size = len(state)
            self.policy = PolicyMLP(self.s_size, self.a_size, self.layers_data).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        if self.use_saved_model:
            self.load_saved_model()

    def convert_state_mlp(self, state):
        # Create array for view data
        view = []
        for i in range(len(state['view'])):
            view = view + state['view'][i]
        cx = state['position'][0]
        cy = state['position'][1]
        cx_min, cx_max = cx-3, cx+3
        cy_min, cy_max = cy-3, cy+3
        lighthouses = np.zeros((7,7), dtype=int)
        lighthouses_dict = dict((tuple(lh["position"]), lh['energy']) for lh in state["lighthouses"])
        for key in lighthouses_dict.keys():
            if cx_min <= key[0] <= cx_max and cy_min <= key[1] <= cy_max:
                lighthouses[key[0]+3-cx, key[1]+3-cy] = lighthouses_dict[key] + 1
        lighthouses_info = []
        # Create array for lighthouses data (within 3 steps of the bot)
        for i in range(len(lighthouses)):
            lighthouses_info = lighthouses_info + list(lighthouses[i])
        new_state = np.array([state['position'][0], state['position'][1], state['score'], state['energy'], len(state['lighthouses'])] + view + lighthouses_info)
        sc = StandardScaler()
        new_state = sc.fit_transform(new_state.reshape(-1, 1))
        new_state = new_state.squeeze()
        return new_state
    
    def z_score_scaling(self, arr):
        arr_mean = np.mean(arr)
        arr_std = np.std(arr)
        scaled_arr = (arr - arr_mean) / arr_std
        return scaled_arr

    def convert_state_cnn(self, state):
        # Create base layer that will serve as the base for all layers of the state
        # This layer has zeros in all cells except the border cells in which the value is -1
        map = np.array(self.map)
        base_layer = np.zeros(map.shape, dtype = int)
        base_layer[0] = -1
        base_layer[len(base_layer)-1] = -1
        base_layer = np.transpose(base_layer)
        base_layer[0] = -1
        base_layer[len(base_layer)-1] = -1
        base_layer = np.transpose(base_layer)

        # Create player layer which has the value of the energy of the player + 1 where the player is located
        # 1 is added to the energy to cover the case that the energy of the player is 0
        player_layer = base_layer.copy()
        x, y = state['position'][0], state['position'][1]
        player_layer[x,y] = 1 + state['energy']
        player_layer = self.z_score_scaling(player_layer)

        # Create view layer with energy level near the player
        view_layer = base_layer.copy()
        state['view'] = np.array(state['view'])
        start_row, start_col = x-3, y-3
        if y+3 > view_layer.shape[1]-1:
            adjust = view_layer.shape[1]-1 - (y+3)
            state['view'] = state['view'][:,:adjust]
        if x+3 > view_layer.shape[0]-1:
            adjust = view_layer.shape[0]-1 - (x+3)
            state['view'] = state['view'][:adjust,:]
        if y-3 < 0:
            adjust = 3-y
            state['view'] = state['view'][:,adjust:]
            start_col = 0
        if x-3 < 0:
            adjust = 3-x
            state['view'] = state['view'][adjust:,:]
            start_row = 0
        view_layer[start_row:start_row+state['view'].shape[0], start_col:start_col+state['view'].shape[1]] = state['view']
        view_layer = self.z_score_scaling(view_layer)

        # Create layer that has the energy of the lighthouse + 1 where the lighthouse is located
        # 1 is added to the lighthouse energy to cover the case that the energy of the lighthouse is 0
        lh_energy_layer = base_layer.copy()
        lh = state['lighthouses']
        for i in range(len(lh)):
            x, y = lh[i]['position'][0], lh[i]['position'][1]
            lh_energy_layer[x,y] = 1 + lh[i]['energy']
        lh_energy_layer = self.z_score_scaling(lh_energy_layer)

        # Create layer that has the number of the layer that controls each lighthouse
        # If no player controls the lighthouse, then a value of -1 is assigned
        lh_control_layer = base_layer.copy()
        for i in range(len(lh)):
            x, y = lh[i]['position'][0], lh[i]['position'][1]
            if not lh[i]['owner']:
                lh[i]['owner'] = -1
            lh_control_layer[x,y] = lh[i]['owner']
        lh_control_layer = self.z_score_scaling(lh_control_layer)

        # Create layer that indicates the lighthouses that are connected
        # If the lighthouse is not connected, then a value of -1 is assigned, if it is connected then it is 
        # assigned the number of connections that it has
        lh_connections_layer = base_layer.copy()
        for i in range(len(lh)):
            x, y = lh[i]['position'][0], lh[i]['position'][1]
            if len(lh[i]['connections']) == 0:
                lh_connections_layer[x,y] = -1
            else:
                lh_connections_layer[x,y] = len(lh[i]['connections'])
        lh_connections_layer = self.z_score_scaling(lh_connections_layer)

        # Create layer that indicates if the player has the key to the light house
        # Assign value of 1 if has key and -1 if does not have key
        lh_key_layer = base_layer.copy()
        for i in range(len(lh)):
            x, y = lh[i]['position'][0], lh[i]['position'][1]
            if lh[i]['have_key']:
                lh_key_layer[x,y] = 1
            else:
                lh_key_layer[x,y] = -1

        # Concatenate the maps into one state
        player_layer = np.expand_dims(player_layer, axis=2)
        view_layer = np.expand_dims(view_layer, axis=2)
        lh_energy_layer = np.expand_dims(lh_energy_layer, axis=2)
        lh_control_layer = np.expand_dims(lh_control_layer, axis=2)
        lh_connections_layer = np.expand_dims(lh_connections_layer, axis=2)
        lh_key_layer = np.expand_dims(lh_key_layer, axis=2)

        new_state = np.concatenate((player_layer, view_layer, lh_energy_layer, lh_control_layer, lh_connections_layer, lh_key_layer), axis=2)
        return new_state
    

    def valid_lighthouse_connections(self, state):
        cx = state['position'][0]
        cy = state['position'][1]
        lighthouses = dict((tuple(lh["position"]), lh) for lh in state["lighthouses"])
        possible_connections = []
        if (cx, cy) in lighthouses:
            if lighthouses[(cx, cy)]["owner"] == self.player_num:
                for dest in lighthouses.keys():
                    if (dest != (cx, cy) and lighthouses[dest]["have_key"] and
                        [cx, cy] not in lighthouses[dest]["connections"] and
                        lighthouses[dest]["owner"] == self.player_num):
                        possible_connections.append(dest)
        return possible_connections


    def play(self, state):
        if self.state_maps:
            print("Using maps for state: PolicyCNN")
            new_state = self.convert_state_cnn(state)
        else:
            print("Using array for state: PolicyMLP")
            new_state = self.convert_state_mlp(state)
        action, log_prob = self.policy.act(new_state)
        self.saved_log_probs.append(log_prob)
        if ACTIONS[action] != "attack" and ACTIONS[action] != "connect" and ACTIONS[action] != "pass":
            return self.move(*ACTIONS[action])
        elif ACTIONS[action] == "pass":
            return self.nop()
        elif ACTIONS[action] == "attack":
            #TODO: improve selection of energy for attacking
            energy = random.randrange(state["energy"] + 1)
            return self.attack(energy)
        elif ACTIONS[action] == "connect":
            # TODO: improve selection of lighthouse connection, right now uses function to select them
            possible_connections = self.valid_lighthouse_connections(state)
            if not possible_connections:
                return self.nop() #pass the turn
            else: 
                return self.connect(random.choice(possible_connections))

    def load_saved_model(self):
        if os.path.isfile(os.path.join(self.model_path, self.model_filename)):
            self.policy.load_state_dict(torch.load(os.path.join(self.model_path, self.model_filename)))
            print("Loaded saved model")
        else:
            print("No saved model")

    def calculate_returns(self, rewards, max_t):
        # Calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards) 
                
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft(self.gamma*disc_return_t + rewards[t])
        # Standardize returns to make training more stable
        eps = np.finfo(np.float32).eps.item()

        # eps is the smallest representable float which is added to the standard
        # deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean())/(returns.std() + eps)

        return returns

    def optimize_model(self, transitions):
        # Calculate the loss and update the policy weights
        rewards = []
        max_t = len(transitions)
        for i in range(max_t):
            rewards.append(transitions[i][2])
        returns = self.calculate_returns(rewards, max_t)

        policy_loss = []
        for log_prob, disc_return in zip(self.saved_log_probs, returns):
            policy_loss.append((-log_prob * disc_return.reshape(1)))
        policy_loss = torch.cat(policy_loss).sum()

        # Calculate the gradient and update the weights
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def save_trained_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(self.model_path, self.model_filename))
        print("Saved model to disk")

    def save_best_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(self.model_path, 'best_'+self.model_filename))
        print("Saved model to disk")