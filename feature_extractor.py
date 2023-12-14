import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.core import ObservationWrapper


# base behavior
class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, regularization = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        if regularization:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
            )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        if regularization:
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2) 
            )
        else:
            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class CustomImgObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict(
            {
                "image": env.observation_space.spaces["image"],
                "vector_pos": env.observation_space.spaces["vector_pos"],
                "vector_info": env.observation_space.spaces["vector_info"]
            }
        )

    def observation(self, obs):
        return {"image": obs["image"], "vector_pos": obs["vector_pos"], "vector_info": obs["vector_info"]}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_spaces: spaces.Dict, cnn_features_dim: int = 512, mlp_features_dim: int = 32, regularization = False) -> None:
        super().__init__(observation_spaces, cnn_features_dim + mlp_features_dim)
        
        # assume observation_spaces["image"] is (3,H,W)
        # assume observation_spaces["vector"] is (D,)

        # n_cnn_input_channels = observation_spaces["image"].shape[0]
        # n_mlp_input_channels = observation_spaces["vector"].shape[0]

        for key, subspace in observation_spaces.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                n_cnn_input_channels = subspace.shape[0]

        if regularization:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_cnn_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(n_cnn_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
            )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_spaces["image"].sample()[None]).float()).shape[1]

        if regularization:
            self.linear = nn.Sequential(
                nn.Linear(n_flatten, cnn_features_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2) 
            )
        else:
            self.linear = nn.Sequential(nn.Linear(n_flatten, cnn_features_dim), nn.ReLU())

        self.mlp = nn.Sequential(
            # can edit this
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, mlp_features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        mlp_input = torch.cat((observations["vector_pos"], observations["vector_info"]), dim = 1)

        encoded_tensor_list.append(self.linear(self.cnn(observations["image"])))
        encoded_tensor_list.append(self.mlp(mlp_input))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)