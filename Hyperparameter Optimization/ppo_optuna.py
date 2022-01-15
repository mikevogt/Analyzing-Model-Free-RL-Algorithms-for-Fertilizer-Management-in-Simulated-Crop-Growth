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

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dim, hl1_dim, hl2_dim, output_dim):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(*input_dim, hl1_dim),
                nn.Tanh(),
                nn.Linear(hl1_dim, hl2_dim),
                nn.Tanh(),
                nn.Linear(hl2_dim, output_dim),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def predict(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist


class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dim, hl1_dim, hl2_dim):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(*input_dim, hl1_dim),
                nn.Tanh(),
                nn.Linear(hl1_dim, hl2_dim),
                nn.Tanh(),
                nn.Linear(hl2_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def predict(self, state):
        value = self.critic(state)

        return value

class PPO:
    def __init__(self, lr, hl1_dim, hl2_dim, gae_lambda,policy_clip, batch_size, n_epochs):
        self.gamma = 1
        self.policy_clip = policy_clip
        self.n_epochs = int(n_epochs)
        self.gae_lambda = gae_lambda
        self.batch_size = int(batch_size)
        self.input_dim = [53]
        self.output_dim = 7

        self.actor = ActorNetwork(lr, self.input_dim, int(hl1_dim), int(hl2_dim), self.output_dim)
        self.critic = CriticNetwork(lr, self.input_dim, int(hl1_dim), int(hl2_dim))
        self.memory = PPOMemory(self.batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor.predict(state)
        value = self.critic.predict(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor.predict(states)
                critic_value = self.critic.predict(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.0888*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

        return critic_loss, actor_loss

def train_ppo_optuna(config, checkpoint_dir=None):
    num_episodes = 5000
    tune.register_env(config["env"], lambda config: FertilizationEnv())
    env = FertilizationEnv()

    fname = 'PPO_' + str(config["lr"]) + '_lr_' + str(config["hl1_dim"]) + '_hl1_dim_' + str(config["hl2_dim"]) + '_hl2_dim_' + \
            str(num_episodes) + '_episodes_' + str(config["gae_lambda"]) + '_gae_lambda_' + \
            str(config["policy_clip"]) + '_policy_clip_' + str(config["batch_size"]) + '_batch_size_' +\
            str(config["N"]) + '_N_' + str(config["n_epochs"]) + '_n_epochs_'
    tb = SummaryWriter(comment=fname)

    agent = PPO(lr=config["lr"], hl1_dim=config["hl1_dim"], hl2_dim=config["hl2_dim"], gae_lambda=config["gae_lambda"], \
                policy_clip=config["policy_clip"], batch_size=config["batch_size"], n_epochs=config["n_epochs"])


    best_score = env.reward_range[0]

    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10., gamma=1)


    learn_iters = 0
    avg_score = 0
    n_steps = 0
    ep_rew=[]

    for i in range(num_episodes):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step([action])
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % config["N"] == 0:
                critic_loss, actor_loss = agent.learn()
                #print(critic_loss)
                tb.add_scalar("critic_loss", critic_loss, i)
                tb.add_scalar("actor_loss", actor_loss, i)

                learn_iters += 1
            observation = observation_

        ep_rew.append(score)
        tb.add_scalar("reward", score, i)
        tune.report(reward=np.mean(ep_rew[-500:]))
    #with tune.checkpoint_dir(step=i) as checkpoint_dir:
    #        path = os.path.join(checkpoint_dir, "checkpoint")
    #        torch.save(agent, path + fname)



if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)

    config = {
        'env': 'fertilization-v0',
        "lr": tune.loguniform(1e-8, 1e-3),
        "hl2_dim": tune.quniform(64,1024,16),
        "hl1_dim": tune.quniform(64,1024,16),
        "gae_lambda": tune.uniform(0.95, 0.99),
        "policy_clip": tune.uniform(0.1, 0.3),
        "batch_size": tune.quniform(64, 640, 32),
        "n_epochs": tune.quniform(1, 10, 1),
        "N": tune.quniform(2, 30, 2)
    }

    init_config = [{
        "lr": 0.0002,
        "hl2_dim": 64,
        "hl1_dim": 64,
        "gae_lambda": 0.98,
        "policy_clip": 0.3,
        "batch_size": 64,
        "n_epochs": 5,
        "N": 1024
    }]

    optuna_search = OptunaSearch(
        metric="reward",
        mode="max",
        points_to_evaluate=init_config)

    optuna_search = ConcurrencyLimiter(optuna_search, max_concurrent=2)

    results = tune.run(
        train_ppo_optuna, # Specify the algorithm to train
        config=config,
        search_alg = optuna_search,
        resources_per_trial={"cpu": 4},
        #verbose=0,
        num_samples=16
    )
    #print(f"Best config: {results.best_config}")
    df = results.dataframe()
    df.to_csv("ppo_optunasearch_8_5000.csv")
