import multiprocessing
import random
from functools import partial
import numpy as np
import torch

from src.environments import PendulumEnv
from src.utils import Discretizer
from src.models import PARAFAC


torch.set_num_threads(1)


gs = [10.0, 10.0, 10.0, 10.0]
ms = [0.01, 0.1, 0.5, 1.0]
ls = [1.0, 1.0, 0.5, 0.5]

envs = [PendulumEnv(g=gs[i], m=ms[i], l=ls[i]) for i in range(len(gs))]

nS = 20
nA = 10
nT = 4

discretizer = Discretizer(
    min_points_states=[-np.pi, -5],
    max_points_states=[np.pi, 5],
    bucket_states=[nS] * 2,
    min_points_actions=[-2],
    max_points_actions=[2],
    bucket_actions=[nA],
)

gamma = 0.99

num_experiments = 100
num_processes = 50
E = 2_000
H = 100
lr = 0.01
eps = 1.0
eps_decay = 0.99999
eps_min = 0.1
k = 30

def create_target(states_next, rewards, Q, tasks=None):
    if tasks is not None:
        idx_target = torch.cat((tasks.unsqueeze(1), states_next), dim=1)
    else:
        idx_target = states_next

    with torch.no_grad():
        q_target = rewards + gamma * Q(idx_target).max(dim=1).values

    return q_target

def create_idx_hat(states, actions, tasks=None):
    if tasks is not None:
        idx_hat = torch.cat((tasks.unsqueeze(1), states, actions.unsqueeze(1)), dim=1)
    else:
        idx_hat = torch.cat((states, actions.unsqueeze(1)), dim=1)
    return idx_hat

def update_model(s_idx, sp_idx, a_idx, r, Q, opt, tasks=None):
    for factor in Q.factors:
        q_target = create_target(sp_idx, r, Q, tasks)
        idx_hat = create_idx_hat(s_idx, a_idx, tasks)
        q_hat = Q(idx_hat)

        opt.zero_grad()
        loss = torch.nn.MSELoss()(q_hat, q_target)
        loss.backward()

        with torch.no_grad():
            for frozen_factor in Q.factors:
                if frozen_factor is not factor:
                    frozen_factor.grad = None

        opt.step()

def select_action(Q, s_idx, epsilon):
    if np.random.rand() < epsilon:
        idx = np.random.choice(nA)
    else:
        with torch.no_grad():
            idx = Q(s_idx).argmax().item()
    
    a = discretizer.get_action_from_index(idx)
    return a, idx

def run_test_episode(Q, env_idx, H):
    with torch.no_grad():
        G = 0
        s, _ = envs[env_idx].reset()
        s_idx = torch.tensor(tuple([env_idx]) + discretizer.get_state_index(s)).unsqueeze(0)
        for h in range(H):
            a_idx = Q(s_idx).argmax().item()
            a = discretizer.get_action_from_index(a_idx)
            a_idx = torch.tensor(a_idx).unsqueeze(0)
            sp, r, d, _, _ = envs[env_idx].step(a)
            sp_idx = torch.tensor(tuple([env_idx]) + discretizer.get_state_index(sp)).unsqueeze(0)

            G += r

            if d:
                break

            s = sp
            s_idx = sp_idx
    return G

def run_experiment(exp_num, E, H, lr, eps, eps_decay, eps_min, k):
    np.random.seed(exp_num)
    random.seed(exp_num)
    torch.manual_seed(exp_num)

    Q = PARAFAC(dims=[nT, nS, nS, nA], k=k, scale=0.1)
    opt = torch.optim.Adamax(Q.parameters(), lr=lr)
    Gs = [[] for _ in range(nT)]
    
    for episode in range(E):
        for env_idx, env in enumerate(envs):
            s, _ = env.reset()
            s_idx = torch.tensor(tuple([env_idx]) + discretizer.get_state_index(s)).unsqueeze(0)
            for h in range(H):
                a, a_idx = select_action(Q, s_idx, eps)
                a_idx = torch.tensor(a_idx).unsqueeze(0)
                sp, r, d, _, _ = env.step(a)
                sp_idx = torch.tensor(tuple([env_idx]) + discretizer.get_state_index(sp)).unsqueeze(0)

                s_idx_train = torch.tensor(discretizer.get_state_index(s)).unsqueeze(0)
                sp_idx_train = torch.tensor(discretizer.get_state_index(sp)).unsqueeze(0)
                update_model(s_idx_train, sp_idx_train, a_idx, r, Q, opt, tasks=torch.tensor([env_idx]))

                if d:
                    break

                s = sp
                s_idx = sp_idx
                eps = max(eps * eps_decay, eps_min)

            G = run_test_episode(Q, env_idx, H)
            Gs[env_idx].append(G)
        print(f"\rEpoch: {episode} - Return: {G}", end="", flush=True)

    return Gs

def pad_Gs(Gs_list):
    max_length = max(len(g) for exp_Gs in Gs_list for g in exp_Gs)
    padded_Gs = np.zeros((len(Gs_list), nT, max_length))

    for i, exp_Gs in enumerate(Gs_list):
        for j, g in enumerate(exp_Gs):
            padded_Gs[i, j, :len(g)] = g

    return padded_Gs

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(partial(run_experiment, E=E, H=H, lr=lr, eps=eps, eps_decay=eps_decay, eps_min=eps_min, k=k), range(num_experiments))

    pool.close()
    pool.join()

    # Pad and save the results
    # padded_Gs = pad_Gs(results)
    np.save("e2_multi.npy", results)
