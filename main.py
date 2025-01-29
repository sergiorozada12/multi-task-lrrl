import argparse
import multiprocessing
import numpy as np
import torch
from src.config.config import ENV_CONFIGS
from src.environments import PendulumEnv, WirelessCommunicationsEnv
from src.models import PARAFAC
from src.rl.experiment_runner import ExperimentRunner

torch.set_num_threads(1)

env_mapping = {"pendulum": PendulumEnv, "wireless": WirelessCommunicationsEnv}

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, choices=["pendulum", "wireless"], required=True)
parser.add_argument("--experiment", type=str, choices=["single", "multi", "single_all"], required=True)
args = parser.parse_args()

envs = [env_mapping[args.env](**params) for params in ENV_CONFIGS[args.env]["env_params"]]
discretizer = ENV_CONFIGS[args.env]["discretizer"]

if __name__ == "__main__":
    experiment = ExperimentRunner(envs, discretizer, PARAFAC, ENV_CONFIGS[args.env], args.experiment)

    with multiprocessing.Pool(processes=ENV_CONFIGS[args.env]["num_processes"]) as pool:
        results = pool.starmap(experiment.run_experiment, [(i,) for i in range(ENV_CONFIGS[args.env]["num_experiments"])])

    np.save(f"results/{args.env}_{args.experiment}.npy", results)
