import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from plot import make_plot

import time
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CustomRewardCallback(BaseCallback):
    def __init__(self, check_freq, reward_threshold, verbose=1):
        super(CustomRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.reward_threshold = reward_threshold
        self.max_reward = 0
        self.episode_rewards = []
        self.mean_rewards = []
        self.fig, self.ax = plt.subplots()
        self.start_time = time.time()  # Start time
        self.threshold_reached_time = None
        self.threshold_reached_steps = None

    def _on_step(self) -> bool:
        # Update the max reward seen so far in the current episode
        self.max_reward = max(self.max_reward, self.locals['rewards'][0])
    
        if self.locals['dones'][0]:
            # Append the maximum reward of the episode
            self.episode_rewards.append(self.max_reward)
            current_mean_reward = np.mean(self.episode_rewards[-100:])
            self.mean_rewards.append(current_mean_reward)

            make_plot(self.episode_rewards, self.mean_rewards, self.fig, self.ax)

            # Reset max_reward for the next episode
            self.max_reward = 0

            if current_mean_reward >= self.reward_threshold and self.threshold_reached_time is None:
                self.threshold_reached_time = time.time() - self.start_time
                self.threshold_reached_steps = self.num_timesteps

                print(f"Threshold reached in {self.threshold_reached_time:.2f} seconds and {self.threshold_reached_steps} steps.")

            if current_mean_reward >= self.reward_threshold:
                print(f"Stopping training as the mean reward {current_mean_reward} is above the threshold {self.reward_threshold}")
                return False  # Return False to stop the training
    
        return True