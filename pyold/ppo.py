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

import argparse
import random
import time
from distutils.util import strtobool

import gym
from torch.utils.tensorboard import SummaryWriter

import random

from bots import bot

# Choose cpu or gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Actions for moving
ACTIONS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1), "attack", "connect", "pass")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_env(seed):
    def thunk():
        env = gym.make("CartPole-v1")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

# Define the neural network for the policy
class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    
# Create training loop
# class PPO(bot.Bot):
class PPO(bot.Bot):
    def __init__(self, seed, num_envs, state_maps=True, model_filename='model.pth', use_saved_model=True):
        super().__init__()
        self.NAME = "REINFORCE"
        self.state_maps = state_maps # use maps for state: True, or array for state: False
        self.a_size = len(ACTIONS)
        self.layers_data = [(64, nn.ReLU()), (64, nn.ReLU())]
        self.gamma = 0.99
        self.learning_rate = 2.5e-4
        self.save_model = True 
        self.model_path = './saved_model'
        self.model_filename = model_filename
        self.use_saved_model = use_saved_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.total_timesteps = 25000
        self.torch_deterministic = True
        self.num_envs = num_envs # the number of steps to run in each environment per policy rollout
        self.anneal_lr = True #learning rate annealing for policy and value networks
        self.gae = True # Use GAE for advantage computation
        self.gae_lambda = 0.95 # lambda for the general advantage estimation
        self.num_minibatches = 4 # the number of mini-batches
        self.update_epochs = 4 # the K epochs to update the policy
        self.norm_adv = True # advantages normalization
        self.clip_coef = 0.2 # the surrogate clipping coefficient
        self.clip_vloss = True # whether or not to use a clipped loss for the value function, as per the paper
        self.ent_coef = 0.01 # coefficient of the entropy
        self.vf_coef = 0.5 # coefficient of the value function
        self.max_grad_norm = 0.5 # maximum norm for the gradient clipping
        self.target_kl = None # the target KL divergence threshold
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        # Initialize tensorboard
        self.writer = SummaryWriter(f"runs/cartpole-test")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self).items()])),
        )

        



    # Initialize agent, optimizer and buffer
    def initialize_game(self, state):
        self.saved_log_probs = []
        if self.state_maps:
            print("using cartpole")
            self.agent = Agent(self.envs).to(device)
            self.initialize_buffer()
            # print("Using maps for state: PolicyCNN")
            # state = self.convert_state_cnn(state)
            # self.num_maps = state.shape[2]
            # self.policy = PolicyCNN(self.num_maps, self.a_size).to(self.device)
        else:
            print("Using array for state: PolicyMLP")
            state = self.convert_state_mlp(state)
            self.s_size = len(state)
            #self.policy = PolicyMLP(self.s_size, self.a_size, self.layers_data).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        if self.use_saved_model:
            self.load_saved_model()
    
    def initialize_buffer(self):
         # ALGO Logic: Storage setup
        self.obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape).to(device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(device)
        # TRY NOT TO MODIFY: start the game
        self.global_step = 0

    
    def get_experiences(self, update):
        start_time = time.time()
        next_obs = torch.Tensor(self.envs.reset()).to(device)
        next_done = torch.zeros(self.num_envs).to(device)
        num_updates = self.total_timesteps // self.batch_size
        # Annealing the rate if instructed to do so.
        if self.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, self.num_steps):
            self.global_step += 1 * self.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
            self.rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={self.global_step}, episodic_return={item['episode']['r']}")
                    self.writer.add_scalar("charts/episodic_return", item["episode"]["r"], self.global_step)
                    self.writer.add_scalar("charts/episodic_length", item["episode"]["l"], self.global_step)
                    break
        
        self.calculate_advantage(next_obs, next_done)


    def calculate_advantage(self, next_obs, next_done):
            # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            if self.gae:
                self.advantages = torch.zeros_like(self.rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                    self.advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                self.returns = self.advantages + self.values
            else:
                self.returns = torch.zeros_like(self.rewards).to(device)
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        next_return = self.returns[t + 1]
                    self.returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
                self.advantages = self.returns - self.values
    
    def optimize_model(self):
        # flatten the batch
        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.agentvalues.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


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
        # possible_connections = [lh["position"] for lh in state["lighthouses"]]
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
            return self.attack(state['energy'])
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

    def save_trained_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(self.model_path, self.model_filename))
        print("Saved model to disk")

    def save_best_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(self.model_path, 'best_'+self.model_filename))
        print("Saved model to disk")

if __name__ == "__main__":
    # Start environment
    seed = 1
    num_envs = 4
    envs = gym.vector.SyncVectorEnv(
    [make_env("CartPole-v1", seed + i, i, "cartpole-test") for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    bot = PPO(seed, num_envs, envs) 

    # Initialize game
    state = None
    self.initialize_game(state)