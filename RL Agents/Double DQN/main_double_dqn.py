import gym
import gym_crop
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

from double_dqn import DDQN

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Double DQN Agent')
    parser.add_argument('-lr', type=float, default=0.00043530, help='learning rate for agent')
    parser.add_argument('-hl1_dim', type=int, default=848, help='hidden layer 1 dimension')
    parser.add_argument('-hl2_dim', type=int, default=672, help='hidden layer 2 dimension')
    parser.add_argument('-gamma', type=float, default=1, help='discount factor for update equation')
    parser.add_argument('-num_episodes', type=int, default=10000, help='number of episodes for agent to play')
    parser.add_argument('-batch_size', type=int, default=544, help='size of mini-batch to sample from replay memory')
    parser.add_argument('-replace', type=int, default=30, help='replace values of target network with values from behaviour network')
    parser.add_argument('-eps_decay', type=float, default=5e-6, help='epsilon decrement amount')
    parser.add_argument('-eps_min', type=float, default=0.03, help='epsilon minimum value')
    parser.add_argument('-extra', type=str, default="", help='extra string to append to save name')
    args = parser.parse_args()

    env = gym.make('fertilization-v0')
    best_score = env.reward_range[0]

    fname = 'Double_DQN_' + str(args.lr) + '_lr_' + str(args.hl1_dim) + '_hl1_dim_' + str(args.hl2_dim) + '_hl2_dims_' + str(args.num_episodes) + '_episodes'\
            + str(args.eps_decay) + '_eps_decay_' + str(args.eps_min) + '_eps_min_' + str(args.replace) + '_repl_' + str(args.batch_size) + '_batch_size' + "_" +args.extra
    tb = SummaryWriter(comment=fname)

    agent = DDQN(lr=args.lr, eps_decay=args.eps_decay, eps_min=args.eps_min, gamma=args.gamma, capacity=20000, batch_size=args.batch_size, replace=args.replace,\
            input_dim=env.observation_space.shape, hl1_dim=args.hl1_dim, hl2_dim=args.hl2_dim, output_dim=env.action_space.n)

    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10., gamma=1)

    n_steps = 0
    t = 0
    for ep in range(args.num_episodes):
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

        tb.add_scalar("reward", score, ep)

        avg_score = np.mean(ep_rew[-100:])
        if avg_score > best_score:
            best_score = avg_score
            torch.save(agent, 'Models/' + fname)

        print('episode ', ep, 'score %.1f' %score, 'avg_score %.1f' %avg_score, 'epsilon %.2f' %agent.epsilon)

    torch.save(agent, 'Models/' + fname + '_end')
