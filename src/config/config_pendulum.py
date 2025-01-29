import numpy as np
from src.utils import Discretizer


PENDULUM_CONFIG = {
    "gamma": 0.99,
    "num_experiments": 100,
    "num_processes": 50,
    "E": 2000,
    "H": 100,
    "lr": 0.01,
    "eps": 1.0,
    "eps_decay": 0.99999,
    "eps_min": 0.1,
    "k": 30,
    "n_upd": 4,

    "env_params": [
        {"g": 10.0, "m": 0.01, "l": 1.0},
        {"g": 10.0, "m": 0.1, "l": 1.0},
        {"g": 10.0, "m": 0.5, "l": 0.5},
        {"g": 10.0, "m": 1.0, "l": 0.5},
    ],

    "discretizer": Discretizer(
        min_points_states=[-np.pi, -5],
        max_points_states=[np.pi, 5],
        bucket_states=[20, 20],
        min_points_actions=[-2],
        max_points_actions=[2],
        bucket_actions=[10],
    )
}
