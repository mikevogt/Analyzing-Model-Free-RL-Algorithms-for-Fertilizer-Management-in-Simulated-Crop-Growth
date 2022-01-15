import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import gym_crop

import os, sys
import pcse
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np

from fertilization_env import FertilizationEnv

import ray
from ray import tune

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.utils.tensorboard import SummaryWriter

import optuna
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest import ConcurrencyLimiter

class DuellingNetwork(nn.Module):
    def __init__(self, lr, input_dim, hl1_dim, hl2_dim, output_dim):
        super(DuellingNetwork, self).__init__()
        self.l1 = nn.Linear(*input_dim, hl1_dim)
        self.l2 = nn.Linear(hl1_dim, hl2_dim)
        self.V = nn.Linear(hl2_dim, 1)
        self.A = nn.Linear(hl2_dim, output_dim)

        #self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def predict(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        V = self.V(x)
        A = self.A(x)
        return V, A

class ReplayMemory():
    def __init__(self, capacity, input_dim):
        self.capacity = capacity
        self.memory_count = 0
        self.state_memory = np.zeros((self.capacity, *input_dim), dtype=np.float32) #, dtype = np.float32
        self.action_memory = np.zeros(self.capacity, dtype=np.int64)
        self.reward_memory = np.zeros(self.capacity, dtype=np.float32)
        self.new_state_memory = np.zeros((self.capacity, *input_dim), dtype=np.float32)
        self.terminal_memory = np.zeros(self.capacity, dtype=np.bool)

    def store(self, state, action, reward, state_, done):
        idx = self.memory_count % self.capacity
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_memory[idx] = state_
        self.terminal_memory[idx] = done
        self.memory_count += 1

    def sample(self, batch_size):
        max_memory = min(self.memory_count, self.capacity)
        batch = np.random.choice(max_memory, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class Duelling_DDQN():
    def __init__(self, lr, eps_decay, batch_size, replace, hl1_dim, hl2_dim):
        self.lr = lr
        self.gamma = 1
        self.input_dim = [53]
        self.hl1_dim = int(hl1_dim)
        self.hl2_dim = int(hl2_dim)
        self.output_dim = 7
        self.action_space = [i for i in range(self.output_dim)]

        self.Q = DuellingNetwork(self.lr, self.input_dim, self.hl1_dim, self.hl2_dim, self.output_dim)
        self.Q_next = DuellingNetwork(self.lr, self.input_dim, self.hl1_dim, self.hl2_dim, self.output_dim)
        self.memory = ReplayMemory(1000000, self.input_dim)

        self.batch_size = int(batch_size)
        self.replace_target_count = int(replace)
        self.learn_step_counter = 0

        self.epsilon = 1
        self.eps_end = 0.03
        self.eps_decay = eps_decay

        #self.loss_history = []
        #self.q_history = []

    def decr_epsilon(self):
        if self.epsilon > self.eps_end:
            self.epsilon = self.epsilon - self.eps_decay
        else:
            self.epsilon = self.eps_end

    def choose_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.Q.device)
        if np.random.random() > self.epsilon:
            _, advantage = self.Q.predict(state)
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_eperience(self, state, action, reward, state_, done):
        self.memory.store(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, state_, done = self.memory.sample(self.batch_size)

        states = torch.tensor(state).to(self.Q.device)
        actions = torch.tensor(action).to(self.Q.device)
        rewards = torch.tensor(reward).to(self.Q.device)
        states_ = torch.tensor(state_).to(self.Q.device)
        dones = torch.tensor(done).to(self.Q.device)

        return states, actions, rewards, states_, dones

    def replace_target(self):
        if self.learn_step_counter % self.replace_target_count == 0:
            self.Q_next.load_state_dict(self.Q.state_dict())

    def learn(self):
        if self.memory.memory_count < self.batch_size:
            return

        self.Q.optimizer.zero_grad()

        self.replace_target()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        V_s, A_s = self.Q.predict(states)
        V_s_, A_s_ = self.Q_next.predict(states_)

        V_s_eval, A_s_eval = self.Q.predict(states_)

        q_pred = torch.squeeze(torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True))))[indices, actions]
        q_next = torch.squeeze(torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))))

        q_eval = torch.squeeze(torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True))))
        max_actions = torch.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = F.mse_loss(q_target, q_pred).to(self.Q.device)

        #q_history.append(q_target)
        #loss_history.append(loss)

        loss.backward()
        self.Q.optimizer.step()
        self.learn_step_counter += 1

        self.decr_epsilon()

        return loss


def train_duelling_ddqn_optuna(config, checkpoint_dir=None):

    num_episodes = 500

    tune.register_env(config["env"], lambda config: FertilizationEnv())
    env = FertilizationEnv()

    agent  = Duelling_DDQN(lr=config["lr"], hl1_dim=config["hl1_dim"], hl2_dim=config["hl2_dim"], \
                 eps_decay=config["eps_decay"], batch_size=config["batch_size"], replace=config["replace"])

    fname = 'Duelling_DDQN_'+ str(config["lr"]) + '_lr_' + str(config["hl1_dim"]) + '_hl1_dim_' + \
                str(config["hl2_dim"]) + '_hl2_dim_' + str(config["batch_size"]) + "_batch_size_" +\
                str(config["replace"]) + "_replace_" + str(config["eps_decay"]) + '_eps_decay_' + str(num_episodes) + '_num_episodes'

    tb = SummaryWriter(comment=fname)

    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10., gamma=1)

    n_steps = 0
    t = 0
    ep_rew=[]
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info = env.step([action])
            score += reward

            agent.store_eperience(state, action, reward, state_, int(done))
            loss = agent.learn()
            if loss == None:
                continue
            else:
                tb.add_scalar("loss", loss, t)
            state = state_
            n_steps += 1
            t += 1
        ep_rew.append(score)
        tb.add_scalar("reward", score, ep)
    tune.report(reward=np.mean(ep_rew[-200:]))
    #torch.save(agent, '/home/mike97vogt/Desktop/Github/crop-gym/gym-crop/gym_crop/envs/Models/DQN_Optuna/v0/' + fname)


if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)

    config = {
        'env': 'fertilization-v0',
        "lr": tune.uniform(1e-8, 1e-3),
        "hl2_dim": tune.quniform(416,992,16),
        "hl1_dim": tune.quniform(416,1024,16),
        "eps_decay": 1e-4,
        "batch_size": tune.quniform(128, 1536, 32),
        "replace": tune.quniform(10, 100, 10)
    }

    init_config = [{
        "lr": 8.99558e-5,
        "hl2_dim":416,
        "hl1_dim":992,
        "batch_size":992,
        "replace" : 80
    }]

    optuna_search = OptunaSearch(
        metric="reward",
        mode="max",
        points_to_evaluate=init_config)

    optuna_search = ConcurrencyLimiter(optuna_search, max_concurrent=1)

    results = tune.run(
        train_duelling_ddqn_optuna, # Specify the algorithm to train
        config=config,
        search_alg = optuna_search,
        resources_per_trial={"cpu": 8},
        #verbose=0,
        num_samples=16
    )
    #print(f"Best config: {results.best_config}")
    df = results.dataframe()
    df.to_csv("double_dqn_optunasearch_64_v1.csv")
