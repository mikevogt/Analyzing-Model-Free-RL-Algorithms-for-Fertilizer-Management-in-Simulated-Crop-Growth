import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, lr, eps_min, eps_decay, gamma, capacity, batch_size, replace, input_dim, hl1_dim, hl2_dim, output_dim):
        self.lr = lr
        self.gamma = gamma
        self.input_dim = input_dim
        self.hl1_dim = hl1_dim
        self.hl2_dim = hl2_dim
        self.output_dim = output_dim
        self.action_space = [i for i in range(self.output_dim)]

        self.Q = DuellingNetwork(self.lr, self.input_dim, self.hl1_dim, self.hl2_dim, self.output_dim)
        self.Q_next = DuellingNetwork(self.lr, self.input_dim, self.hl1_dim, self.hl2_dim, self.output_dim)
        self.memory = ReplayMemory(capacity, self.input_dim)

        self.batch_size = batch_size
        self.replace_target_count = replace
        self.learn_step_counter = 0

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
            _, advantage = self.Q.predict(state)
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def predict_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.Q.device)
        _ , actions = self.Q.predict(state)
        action = torch.argmax(actions).item()
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
