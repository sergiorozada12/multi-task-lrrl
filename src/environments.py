from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class PendulumEnv(gym.Env):
    def __init__(self, g=10.0, m=1.0, l=1.0):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = m
        self.l = l

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, u):
        th, thdot = self.state

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.1 * (u ** 2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        return self._get_obs(), -costs, False, False, {}

    def reset(self, *, seed: Optional[int] = None):
        super().reset(seed=seed)
        self.state = [np.random.rand()/100, np.random.rand()/100]
        self.last_u = None
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot], dtype=np.float32)

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
