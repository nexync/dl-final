import torch
import torch.nn as nn
import gymnasium as gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.core import ObservationWrapper


# base behavior
class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
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

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class CustomImgObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        return {"image": obs["image"], "vector": obs["vector"]}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_spaces: gym.spaces.Dict, cnn_features_dim: int = 512, mlp_features_dim: int = 32) -> None:
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
            elif key == "vector":
                # Run through a simple MLP
                n_mlp_input_channels = subspace.shape[0]

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
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, cnn_features_dim), nn.ReLU()
        )

        self.mlp = nn.Sequential(
            # can edit this
            nn.Linear(n_mlp_input_channels, 32),
            nn.ReLu(),
            nn.Linear(32, mlp_features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key in observations:
            if key == "image":
                encoded_tensor_list.append(self.linear(self.cnn(observations[key])))
            elif key == "vector":
                encoded_tensor_list.append(self.mlp(observations[key]))
            else:
                assert False

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)