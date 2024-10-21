from typing import Optional, List

import numpy as np

import gymnasium as gym
from gymnasium import spaces


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


"""
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
        # costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.1 * (u ** 2)
        costs = (u ** 2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        done = np.abs(newth) > (np.pi / 4)
        costs += 1_000 * done

        self.state = np.array([newth, newthdot])

        return self._get_obs(), -costs, done, False, {}

    def reset(self, *, seed: Optional[int] = None):
        super().reset(seed=seed)
        #self.state = [np.random.rand()/100, np.random.rand()]
        self.state = [0.0, np.random.rand()]
        self.last_u = None
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot], dtype=np.float32)"""


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

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        newth = (newth + np.pi) % (2 * np.pi) - np.pi

        cost = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.1 * (u ** 2)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -cost, False, False, {}


    def reset(self, *, seed: Optional[int] = None):
        super().reset(seed=seed)
        pin = np.pi
        win = np.random.rand() / 10
        self.state = [pin, win]
        self.last_u = None
        return self._get_obs(), {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot], dtype=np.float32)

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class WirelessCommunicationsEnv:
    """
    SETUP DESCRIPTION
    - Wireless communication setup, focus in one user sharing the media with other users
    - Finite horizon transmission (T slots)
    - User is equipped with a battery and queue where info is stored
    - K orthogonal channels, user can select power for each time instant
    - Rate given by Shannon's capacity formula

    STATES
    - Amount of energy in the battery: bt (real and positive)
    - Amount of packets in the queue: queuet (real and positive)
    - Normalized SNR (aka channel gain) for each of the channels: gkt (real and positive)
    - Channel being currently occupied: ok (binary)

    ACTIONS
    - Accessing or not each of the K channels pkt
    - Tx power for each of the K channels
    - We can merge both (by setting pkt=0)
    """

    def __init__(
        self,
        T: int = 10,  # Number of time slots
        K: int = 3,  # Number of channels
        snr_max: float = 10,  # Max SNR
        snr_min: float = 2,  # Min SNR
        snr_autocorr: float = 0.7,  # Autocorrelation coefficient of SNR
        P_occ: np.ndarray = np.array(
            [  # Prob. of transition of occupancy
                [0.3, 0.5],
                [0.7, 0.5],
            ]
        ),
        occ_initial: List[int] = [1, 1, 1],  # Initial occupancy state
        batt_harvest: float = 3,  # Battery to harvest following a Bernoulli
        P_harvest: float = 0.5,  # Probability of harvest energy
        batt_initial: float = 5,  # Initial battery
        batt_max_capacity: float = 50,  # Maximum capacity of the battery
        batt_weight: float = 1.0,  # Weight for the reward function
        queue_initial: float = 20,  # Initial size of the queue
        queue_arrival: float = 20, # Arrival messages
        queue_max_capacity: float = 20,
        t_queue_arrival: int = 10, # Refilling of the queue
        queue_weight: float = 1e-1,  # Weight for the reward function
        loss_busy: float = 0.80,  # Loss in the channel when busy
        n_packets: int = 1,
    ) -> None:
        self.T = T
        self.K = K

        self.snr = np.linspace(snr_max, snr_min, K)
        self.snr_autocorr = snr_autocorr

        self.occ_initial = occ_initial
        self.P_occ = P_occ

        self.batt_harvest = batt_harvest
        self.batt_initial = batt_initial
        self.P_harvest = P_harvest
        self.batt_max_capacity = batt_max_capacity
        self.batt_weight = batt_weight

        self.queue_initial = queue_initial
        self.queue_weight = queue_weight
        self.t_queue_arrival = t_queue_arrival
        self.queue_arrival = queue_arrival
        self.queue_max_capacity = queue_max_capacity

        self.loss_busy = loss_busy
        self.n_packets = n_packets

    def step(self, p: np.ndarray):
        p = np.clip(p, 0, 2)
        if np.sum(p) > self.batt[self.t]:
            p = self.batt[self.t] * p / np.sum(p)

        self.c[:, self.t] = np.log2(1 + self.g[:, self.t] * p)
        self.c[:, self.t] *= (1 - self.loss_busy) * self.occ[:, self.t] + (
            1 - self.occ[:, self.t]
        )

        self.t += 1

        self.h[:, self.t] = np.sqrt(0.5 * self.snr) * (
            np.random.randn(self.K) + 1j * np.random.randn(self.K)
        )
        self.h[:, self.t] *= np.sqrt(1 - self.snr_autocorr)
        self.h[:, self.t] += np.sqrt(self.snr_autocorr) * self.h[:, self.t - 1]
        self.g[:, self.t] = np.abs(self.h[:, self.t]) ** 2

        # self.P_occ[1, 1] -> prob getting unocc
        self.occ[:, self.t] += (np.random.rand(self.K) > self.P_occ[1, 1]) * self.occ[
            :, self.t - 1
        ]

        # self.P_occ[0, 0] -> prob keeping unocc
        self.occ[:, self.t] += (np.random.rand(self.K) > self.P_occ[0, 0]) * (
            1 - self.occ[:, self.t - 1]
        )

        energy_harv = self.batt_harvest * (self.P_harvest > np.random.rand())
        self.batt[self.t] = self.batt[self.t - 1] - np.sum(p) + energy_harv
        self.batt[self.t] = np.clip(self.batt[self.t], 0, self.batt_max_capacity)

        packets = 0
        if self.batt[self.t - 1] > 0:
            packets = np.sum(self.c[:, self.t - 1])
        self.queue[self.t] = self.queue[self.t - 1] - packets

        # if self.t % self.t_queue_arrival == 0:
            # self.queue[self.t] += self.queue_arrival
        self.queue[self.t] += self.n_packets if np.random.rand() < 0.5 else 0

        self.queue[self.t] = np.clip(self.queue[self.t], 0, self.queue_max_capacity)

        r = (self.batt_weight * np.log(1 + self.batt[self.t]) - self.queue_weight * self.queue[self.t])
        done = self.t == self.T

        return self._get_obs(self.t), r, done, None, None

    def reset(self):
        self.t = 0
        self.h = np.zeros((self.K, self.T + 1), dtype=np.complex64)
        self.g = np.zeros((self.K, self.T + 1))
        self.c = np.zeros((self.K, self.T + 1))
        self.occ = np.zeros((self.K, self.T + 1))
        self.queue = np.zeros(self.T + 1)
        self.batt = np.zeros(self.T + 1)

        self.h[:, 0] = np.sqrt(0.5 * self.snr) * (
            np.random.randn(self.K) + 1j * np.random.randn(self.K)
        )
        self.g[:, 0] = np.abs(self.h[:, 0]) ** 2
        self.occ[:, 0] = self.occ_initial
        self.queue[0] = self.queue_initial
        self.batt[0] = self.batt_initial

        return self._get_obs(0), None

    def _get_obs(self, t):
        return np.concatenate(
            [self.g[:, t], self.occ[:, t], [self.queue[t], self.batt[t]]]
        )
