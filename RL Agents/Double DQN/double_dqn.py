import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, lr, input_dim, hl1_dim, hl2_dim, output_dim):
        super(Network, self).__init__()
        self.l1 = nn.Linear(*input_dim, hl1_dim)
        self.l2 = nn.Linear(hl1_dim, hl2_dim)
        self.l3 = nn.Linear(hl2_dim, output_dim)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def predict(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

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

class DDQN():
    def __init__(self, lr, eps_decay, eps_min, gamma, capacity, batch_size, replace, input_dim, hl1_dim, hl2_dim, output_dim): # maybe replace
        self.lr = lr
        self.gamma = gamma
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_space = [i for i in range(self.output_dim)]

        self.batch_size = batch_size
        self.replace_target_count = replace
        self.learn_step_counter = 0

        self.Q = Network(self.lr, self.input_dim, hl1_dim, hl2_dim, self.output_dim)
        self.Q_next = Network(self.lr, self.input_dim, hl1_dim, hl2_dim, self.output_dim)
        self.memory = ReplayMemory(capacity, input_dim)

        self.epsilon = 1
        self.eps_end = eps_min
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
            actions = self.Q.predict(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def predict_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.Q.device)
        actions = self.Q.predict(state)
        #print(actions)
        action = torch.argmax(actions).item()
        return action

    def store_eperience(self, state, action, reward, state_, done):
        self.memory.store(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, state_, done = self.memory.sample(self.batch_size)

        states = torch.tensor(np.array([state]), dtype=torch.float).to(self.Q.device)
        actions = torch.tensor(action).to(self.Q.device)
        rewards = torch.tensor(reward).to(self.Q.device)
        states_ = torch.tensor(np.array([state_]), dtype=torch.float).to(self.Q.device)
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

        q_val = torch.squeeze(self.Q.predict(states))[indices, actions]
        #print("q_val ", q_val.shape)
        q_next = torch.squeeze(self.Q_next.predict(states_))
        #print("q_next ", q_next.shape)
        q_eval = torch.squeeze(self.Q.predict(states_))
        #print("q_eval ", q_eval.shape)
        max_actions = torch.argmax(q_eval, dim=1)
        #print('max_actions: ', max_actions.shape)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.Q.loss(q_target, q_val).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()

        self.learn_step_counter += 1
        self.decr_epsilon()

        return loss
