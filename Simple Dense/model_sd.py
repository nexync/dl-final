#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym

###############################Simple Dense######################

class SimpleDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return F.softmax(x, dim=-1)
    
    def save(self, file_name='model_sd.pth'):
        model_path = './model_sd'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        file_name = os.path.join(model_path, file_name)
        torch.save(self.state_dict(), file_name)

#####################Extractor##########################
# class MinigridFeaturesExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None:
#         super().__init__(observation_space, features_dim)
#         n_input_channels = observation_space.shape[0]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Compute shape by doing one forward pass
#         with torch.no_grad():
#             n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

#         self.linear = nn.Sequential(
#             nn.Linear(n_flatten, features_dim),
#             nn.ReLU(),
#             nn.Dropout(p=0.5), 
#         )

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         if observations.max() > 1.0:
#             observations = observations / 255.0  
#         return self.linear(self.cnn(observations))

        
#####################Image to Action#####################

# class img2act(nn.Module):
#     def __init__(self, h, w, outputs):
#         super(SimpleDense, self).__init__()
#         self.conv1 = nn.Conv2d(2, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)


#         def conv2d_size_out(size, kernel_size = 5, stride = 2):
#             return (size - (kernel_size - 1) - 1) // stride  + 1
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
#         linear_input_size = convw * convh * 32
#         self.head = nn.Linear(linear_input_size, outputs)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = x.view(x.size(0), -1)

#         return self.head(x)
    
#     def save(self, file_name2='img2act.pth'):
#         model_path2 = './img2act'
#         if not os.path.exists(model_path2):
#             os.makedirs(model_path2)

#         file_name2 = os.path.join(model_path2, file_name2)
#         torch.save(self.state_dict(), file_name2)


###################Trainer#######################

class Trainer:
    def __init__(self, model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=20)
        self.criterion = nn.SmoothL1Loss() # nn.MSELoss() 

    def train_step(self, states, actions, rewards, next_states, dones):
        # Convert states and actions to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Predicted Q values
        pred = self.model(states)

        # Handle next_states that are None
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=self.device, dtype=torch.bool)
        non_final_next_states = [s for s in next_states if s is not None]
        if non_final_next_states:
            non_final_next_states = torch.tensor(np.array(non_final_next_states), dtype=torch.float32, device=self.device)
            next_state_values = torch.zeros(len(next_states), device=self.device)
            next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]
        else:
            next_state_values = torch.zeros(len(next_states), device=self.device)

        # Compute the target Q values
        target_Q_values = rewards + (self.gamma * next_state_values * (1 - dones))

        # Gather only the Q values corresponding to the taken actions
        action_Q_values = pred.gather(1, actions.unsqueeze(1))

        # Calculate loss
        self.optimizer.zero_grad()
        loss = self.criterion(action_Q_values, target_Q_values.unsqueeze(1))
        loss.backward()
        self.optimizer.step()
        
    def update_learning_rate(self, avg_reward):
        self.scheduler.step(avg_reward)

