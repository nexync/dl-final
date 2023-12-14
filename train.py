import argparse
from pathlib import Path

from stable_baselines3 import PPO

from feature_extractor import CNNFeaturesExtractor, CustomFeatureExtractor, CustomImgObsWrapper
from minigrid.wrappers import ImgObsWrapper
from callback import CustomRewardCallback

from env import CustomDoorKey

def parse_args():
	"""
		Parse command-line arguments
	"""
	parser = argparse.ArgumentParser(description="Train a PPO model on Doorkey Environment")

	parser.add_argument(
		"-s",
		"--size",
		type = int,
		help = "Size of Doorkey environment (default is 8)",
		default = 8,
	)

	parser.add_argument(
		"-r",
		"--regularization",
		action = "store_true",
		help = "Use regularization in feature extractor",
		default = True,
	)

	parser.add_argument(
		"-m",
		"--multi",
		action = "store_true",
		help = "Use multi-input for feature extractor",
		default = False,
	)

	parser.add_argument(
		"-m",
		"--model",
		type = str,
		help = "Optional trained agent to load (for further training)"
	)

	parser.add_argument(
		"-v",
		"--verbose",
		action = "store_true",
		help = "Show graph during training",
		default = False,
	)

	parser.add_argument(
		"--max_iters",
		type = int,
		default = 3e5,
		help = "Maximum amount of timesteps allowed during training"
	)

	parser.add_argument(
		"--threshold",
		type = float,
		default = 0.9,
		help = "Threshold value at which to stop training"
	)

	parser.add_argument(
		"--save_path",
		type = str,
		default = "trained_agent",
		help = "Path to save trained model"
	)

	args = parser.parse_args()
	return args

def main() -> None:
	args = parse_args()

	env = CustomDoorKey(size=args.size, intermediate_reward=True, randomize_goal=True, render_mode = "rgb")
	
	if args.multi:
		env = CustomImgObsWrapper(env)

		policy_kwargs = dict(
			features_extractor_class=CustomFeatureExtractor,
			features_extractor_kwargs=dict(cnn_features_dim=128, mlp_features_dim=32, regularization = args.regularization),
		)

	else:
		env = ImgObsWrapper(env)

		policy_kwargs = dict(
			features_extractor_class=CNNFeaturesExtractor,
			features_extractor_kwargs=dict(features_dim=128, regularization = args.regularization),
		)

	callback = CustomRewardCallback(check_freq=1000, reward_threshold=args.threshold, verbose=args.verbose)  # set callback

	if args.model is not None:
		model = PPO.load(Path(args.model), env = env)
	else:
		if args.multi:
			model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=False)
		else:
			model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=False)

	model.learn(args.max_iters, callback=callback)
	model.save(args.save_path)

if __name__ == "__main__":
	main()