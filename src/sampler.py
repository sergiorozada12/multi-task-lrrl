import numpy as np
import torch
from torch.utils.data import Dataset

from src.environments import PendulumEnv
from src.utils import Discretizer


class PendulumTrajectorySampler:
    def __init__(self, gs, m, l, E, H, nS, nA):
        self.gs = gs
        self.m = m
        self.l = l
        self.E = E
        self.H = H
        self.nS = nS
        self.nA = nA

        self.tasks, self.states, self.states_next, self.actions, self.rewards = [], [], [], [], []

    def sample_data(self):
        for i, g in enumerate(self.gs):
            env = PendulumEnv(g, self.m, self.l)

            discretizer = Discretizer(
                min_points_states=[-1, -5],
                max_points_states=[1, 5],
                bucket_states=[self.nS] * 2,
                min_points_actions=[-2],
                max_points_actions=[2],
                bucket_actions=[self.nA],
            )

            for _ in range(self.E):
                s, _ = env.reset()
                s_idx = discretizer.get_state_index(s)
                for _ in range(self.H):
                    a_idx = np.random.choice(self.nA)
                    a = discretizer.get_action_from_index(a_idx)
                    sp, r, d, _, _ = env.step(a)
                    sp_idx = discretizer.get_state_index(sp)

                    self.states.append(s_idx)
                    self.states_next.append(sp_idx)
                    self.actions.append(a_idx)
                    self.rewards.append(r)
                    self.tasks.append(i)

                    if d:
                        break

                    s = sp
                    s_idx = sp_idx

        self.tasks = torch.tensor(self.tasks)
        self.states = torch.tensor(self.states)
        self.states_next = torch.tensor(self.states_next)
        self.actions = torch.tensor(self.actions)
        self.rewards = torch.tensor(self.rewards)

    def get_trajectories_dataset(self, task_id=None):
        return Trajectories(self.tasks, self.states, self.states_next, self.actions, self.rewards, task_id)


class Trajectories(Dataset):
    def __init__(self, tasks, states, states_next, actions, rewards, task_id=None):
        self.tasks = tasks
        self.states = states
        self.states_next = states_next
        self.actions = actions
        self.rewards = rewards

        self.task_id = task_id
        if task_id is not None:
            mask = (self.tasks == task_id)
            self.tasks = self.tasks[mask]
            self.states = self.states[mask]
            self.states_next = self.states_next[mask]
            self.actions = self.actions[mask]
            self.rewards = self.rewards[mask]

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        if self.task_id is not None:
            return self.states[idx], self.states_next[idx], self.actions[idx], self.rewards[idx]
        return self.tasks[idx], self.states[idx], self.states_next[idx], self.actions[idx], self.rewards[idx]
