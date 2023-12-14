#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

import matplotlib.pyplot as plt
import math
import random
from itertools import count
from collections import namedtuple
from collections import deque

from env import myenv
from plot import make_plot
from model_sd import SimpleDense, Trainer


from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

#%%
Capacity = 100000
BATCH_SIZE = 1000
LR = 1

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Agent:
    def __init__(self, action_space, epsilon_decay=0.95, preprocess_obss=None, use_memory=False):
        self.gamma = 0.9 # discount rate
        self.model = SimpleDense() # Model 
        
        self.action_space = action_space
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        
        self.preprocess_obss = preprocess_obss  
        self.use_memory = use_memory
        memory_size = 128 # If use RNN arch
        self.memory = deque(maxlen=Capacity) 
        self.trainer = Trainer(self.model, lr=LR, gamma=self.gamma, device = device)
        if use_memory:
            self.memories = torch.zeros(1, memory_size, device=device)
            
    def get_state(self, env):
       # Get the agent's position and direction
       agent_pos = env.agent_pos
       agent_dir = env.agent_dir

       # Get key and door positions, and door state
       key_pos = None
       door_pos = None
       door_open = False

       # Search the grid for the key and the door
       for x in range(env.width):
           for y in range(env.height):
               cell = env.grid.get(x, y)
               if isinstance(cell, Key): 
                   key_pos = (x, y)
               elif isinstance(cell, Door):  
                   door_pos = (x, y)
                   door_open = cell.is_open

       # if key or door is not found
       key_pos = key_pos if key_pos is not None else (-1, -1)
       door_pos = door_pos if door_pos is not None else (-1, -1)

       # State representation
       state = [
           agent_pos[0], agent_pos[1],  
           agent_dir,
           key_pos[0], key_pos[1],
           door_pos[0], door_pos[1],
           int(door_open)
       ]

       return np.array(state)

    def get_action(self, obs):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)
        else:
            # Existing logic for choosing the best action
            if self.preprocess_obss:
                obs = self.preprocess_obss(obs)

            obs = torch.tensor(obs, dtype=torch.float).to(device).unsqueeze(0)
            with torch.no_grad():
                if self.use_memory and self.memories is not None:
                    action_probs, self.memories = self.model(obs, self.memories)
                else:
                    action_probs = self.model(obs)

            action = torch.argmax(action_probs).item()
            return action
        
    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 100000:
            self.memory.popleft()

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step([state], [action], [reward], [next_state], [done])
        
    def analyze_feedback(self, reward, done):
        if self.use_memory and self.memories is not None:
            mask = 1 - torch.tensor(done, dtype=torch.float).unsqueeze(0)
            self.memories *= mask
            
# Training Loop
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    max_steps_per_episode = 100
    env = myenv(render_mode="human")
    action_space = env.action_space.n
    agent = Agent(action_space=action_space)

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.ion()  # Turn on interactive mode
    
    max_episodes = 1000
    agent.n_trial = 0
    total_reward = 0
    
    while agent.n_trial < max_episodes:
        obs = env.reset()
        state_old = agent.get_state(env)
        score = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            action = agent.get_action(state_old)
            obs_dict, reward, done, extra_flag, info = env.step(action)  
            obs = obs_dict['image']
            env.render()

            score += reward
            
            state_new = agent.get_state(env) if not done else None

            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)
            state_old = state_new
            
            step_count += 1
            
        total_reward += score
        # learning rate scheduler
        avg_reward = total_reward / (agent.n_trial + 1)
        agent.trainer.update_learning_rate(avg_reward)

        agent.n_trial += 1
        agent.train_long_memory()

        if score > record:
            record = score
            agent.model.save()

        print('Episode', agent.n_trial, 'Score', score, 'Record:', record)
        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_trial
        plot_mean_scores.append(mean_score)
        make_plot(plot_scores, plot_mean_scores, fig, ax) 

    plt.ioff()  # Turn off interactive mode

if __name__ == '__main__':
    train()
    

# %%
