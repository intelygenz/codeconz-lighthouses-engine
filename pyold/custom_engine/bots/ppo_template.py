#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules import Module
from torch.distributions import Categorical
from sklearn.preprocessing import StandardScaler

import random
import time
from distutils.util import strtobool

import gym
from torch.utils.tensorboard import SummaryWriter

import random

# from bots import bot

# Choose cpu or gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
class PPO():
    def __init__(self, seed, num_envs, envs, num_steps, num_updates, model_filename='model.pth', use_saved_model=False):
        super().__init__()
        self.NAME = "REINFORCE"
        self.gamma = 0.99
        self.learning_rate = 2.5e-4
        self.save_model = True 
        self.model_path = './saved_model'
        self.model_filename = model_filename
        self.use_saved_model = use_saved_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.torch_deterministic = True
        self.envs = envs
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
        self.num_steps = num_steps
        self.num_updates = num_updates
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
        print("using cartpole")
        self.agent = Agent(self.envs).to(device)
        self.initialize_buffer_and_variables()
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)
        if self.use_saved_model:
            self.load_saved_model()
    

    def initialize_buffer_and_variables(self):
         # ALGO Logic: Storage setup
        self.obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape).to(device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.global_step = 0
        self.start_time = time.time()
        # self.next_obs = torch.Tensor(self.envs.reset()).to(device)
        # self.next_done = torch.zeros(self.num_envs).to(device)


    def initialize_experience_gathering(self, update):
        # Annealing the rate if instructed to do so.
        if self.anneal_lr:
            frac = 1.0 - (update - 1.0) / self.num_updates
            lrnow = frac * self.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow
    
    def play(self, next_obs, next_done):
        self.global_step += 1 * self.num_envs
        self.obs[step] = next_obs
        self.dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(next_obs)
            self.values[step] = value.flatten()
        self.actions[step] = action
        self.logprobs[step] = logprob

        return action
    
    def print_metrics(self, info):
        for item in info:
            if "episode" in item.keys():
                print(f"global_step={self.global_step}, episodic_return={item['episode']['r']}")
                self.writer.add_scalar("charts/episodic_return", item["episode"]["r"], self.global_step)
                self.writer.add_scalar("charts/episodic_length", item["episode"]["l"], self.global_step)
                break


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
    
    def optimize_model(self, next_obs, next_done):
        # flatten the batch
        self.calculate_advantage(next_obs, next_done)
        b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

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
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
        print("SPS:", int(self.global_step / (time.time() - self.start_time)))
        self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)


    

if __name__ == "__main__":
    # Start environment
    SEED = 1
    NUM_ENVS = 4
    TOTAL_TIMESTEPS = 25000
    NUM_STEPS = 128
    NUM_UPDATES = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)

    envs = gym.vector.SyncVectorEnv(
    [make_env(SEED + i) for i in range(NUM_ENVS)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    bot = PPO(seed=SEED, num_envs=NUM_ENVS, envs=envs, num_steps=NUM_STEPS, num_updates=NUM_UPDATES)

    # Initialize game
    state = None
    bot.initialize_game(state)
    state = torch.Tensor(envs.reset()).to(device)
    done = torch.zeros(NUM_ENVS).to(device)

    for update in range(1, NUM_UPDATES + 1):
        bot.initialize_experience_gathering(update)
        for step in range(0, NUM_STEPS):
            action = bot.play(state, done)
            # TRY NOT TO MODIFY: execute the game and log data.
            next_state, reward, done, info = envs.step(action.cpu().numpy())
            bot.rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_state, done = torch.Tensor(next_state).to(device), torch.Tensor(done).to(device)
            bot.print_metrics(info)
            state = next_state
        
        bot.optimize_model(state, done)
