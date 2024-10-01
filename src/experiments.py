import multiprocessing
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.sampler import PendulumTrajectorySampler
from src.trainer import QNetworkTrainer, QNetworkTester
from src.models import PARAFAC


torch.set_num_threads(1)


class ExperimentRunner:
    def __init__(self, gs, m, l, E, H, nS, nA, k, lr, gamma, epochs, n_exps):
        self.gs = gs
        self.m = m
        self.l = l
        self.E = E
        self.H = H
        self.nS = nS
        self.nA = nA
        self.k = k
        self.lr = lr
        self.gamma = gamma
        self.epochs = epochs
        self.n_exps = n_exps
        self.nT = len(gs)

    def run_single_task_experiment(self, n, task_id=0):
        np.random.seed(n)
        random.seed(n)
        torch.manual_seed(n)

        sampler = PendulumTrajectorySampler(self.gs, self.m, self.l, self.E, self.H, self.nS, self.nA)
        sampler.sample_data()

        dataset_single = sampler.get_trajectories_dataset(task_id=task_id)
        loader_single = DataLoader(dataset_single, batch_size=128, shuffle=True)

        Q = PARAFAC(dims=[self.nS, self.nS, self.nA], k=self.k, scale=0.1)
        trainer = QNetworkTrainer(Q, self.lr, self.gamma)
        trainer.train(loader_single, self.epochs, use_tasks=False)
        tester = QNetworkTester(Q, self.nS, self.nA, self.gs, self.m, self.l, self.H)
        return tester.test(task_id=task_id, multi_task=False)

    def run_single_all_task_experiment(self, n):
        np.random.seed(n)
        random.seed(n)
        torch.manual_seed(n)

        sampler = PendulumTrajectorySampler(self.gs, self.m, self.l, self.E, self.H, self.nS, self.nA)
        sampler.sample_data()

        dataset_mult = sampler.get_trajectories_dataset()
        loader_mult = DataLoader(dataset_mult, batch_size=128, shuffle=True)

        Q = PARAFAC(dims=[self.nS, self.nS, self.nA], k=self.k, scale=0.1)
        trainer = QNetworkTrainer(Q, self.lr, self.gamma)
        trainer.train(loader_mult, self.epochs, use_tasks=False)
        tester = QNetworkTester(Q, self.nS, self.nA, self.gs, self.m, self.l, self.H)
        return tester.test(task_id=None, multi_task=False)

    def run_multi_task_experiment(self, n):
        np.random.seed(n)
        random.seed(n)
        torch.manual_seed(n)

        sampler = PendulumTrajectorySampler(self.gs, self.m, self.l, self.E, self.H, self.nS, self.nA)
        sampler.sample_data()

        dataset_mult = sampler.get_trajectories_dataset()
        loader_mult = DataLoader(dataset_mult, batch_size=128, shuffle=True)

        Q = PARAFAC(dims=[self.nT, self.nS, self.nS, self.nA], k=self.k, scale=0.1)
        trainer = QNetworkTrainer(Q, self.lr, self.gamma)
        trainer.train(loader_mult, self.epochs, use_tasks=True)
        tester = QNetworkTester(Q, self.nS, self.nA, self.gs, self.m, self.l, self.H)
        return tester.test(task_id=None, multi_task=True)

    def run_experiments_distributed(self, experiment_type, task_id=None):
        if experiment_type == 'single_task':
            experiment_function = partial(self.run_single_task_experiment, task_id=task_id)
        elif experiment_type == 'single_all_task':
            experiment_function = self.run_single_all_task_experiment
        elif experiment_type == 'multi_task':
            experiment_function = self.run_multi_task_experiment
        else:
            raise ValueError("Invalid experiment type")

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(experiment_function, range(self.n_exps))
        return results
