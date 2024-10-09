import multiprocessing
import random
from functools import partial
import numpy as np
import torch

from src.environments import WirelessCommunicationsEnv
from src.utils import Discretizer
from src.models import PARAFAC

torch.set_num_threads(1)


H = 100

ts = [20, 20, 20, 20]
p_harv = [0.5, 0.5, 0.5, 0.5]
qarr = [3, 3, 2, 1]
lbus = [1.0, 0.8, 0.8, 0.1]

envs = [WirelessCommunicationsEnv(
    T=H,
    K=2,
    snr_max=10,
    snr_min=8,
    snr_autocorr=0.7,
    P_occ=np.array(
        [  
            [0.4, 0.6],
            [0.6, 0.4],
        ]
    ),
    occ_initial=[1, 1],
    batt_harvest=1.0, 
    P_harvest=0.2, 
    batt_initial=10,
    batt_max_capacity=10,
    batt_weight=1.0, 
    queue_initial=5,
    queue_arrival=qarr[i],
    queue_max_capacity=20,
    t_queue_arrival=ts[i],
    queue_weight=0.2,
    loss_busy=lbus[i],  
) for i in range(len(ts))]

nS = [20, 20, 2, 2, 20, 20]
nA = [10, 10]
nT = len(ts)
gamma = 0.99

discretizer = Discretizer(
    min_points_states=[0, 0, 0, 0, 0, 0],
    max_points_states=[20, 20, 1, 1, 20, 10],
    bucket_states=[20, 20, 2, 2, 20, 20],
    min_points_actions=[0, 0],
    max_points_actions=[2, 2],
    bucket_actions=[10, 10],
)

num_experiments = 100
num_processes = 50
E = 2000
lr = 0.01
eps = 1.0
eps_decay = 0.99999
eps_min = 0.01
k = 20

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
        idx_hat = torch.cat((tasks.unsqueeze(1), states, actions), dim=1)
    else:
        idx_hat = torch.cat((states, actions), dim=1)
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

def select_random_action() -> np.ndarray:
        a_idx = tuple(np.random.randint(discretizer.bucket_actions).tolist())
        return discretizer.get_action_from_index(a_idx), a_idx

def select_greedy_action(Q, s_idx: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        a_idx_flat = Q(s_idx).argmax().detach().item()
        a_idx = np.unravel_index(a_idx_flat, discretizer.bucket_actions)
        return discretizer.get_action_from_index(a_idx), a_idx

def select_action(Q, s_idx: np.ndarray, epsilon: float) -> np.ndarray:
    if np.random.rand() < epsilon:
        return select_random_action()
    return select_greedy_action(Q, s_idx)

def run_test_episode(Q, env_idx):
    with torch.no_grad():
        G = 0
        s, _ = envs[env_idx].reset()
        s_idx = torch.tensor(discretizer.get_state_index(s)).unsqueeze(0)
        for h in range(H):
            a_idx = Q(s_idx).argmax().item()
            a = discretizer.get_action_from_index(a_idx)
            a_idx = torch.tensor(a_idx).unsqueeze(0)
            sp, r, d, _, _ = envs[env_idx].step(a)
            sp_idx = torch.tensor(discretizer.get_state_index(sp)).unsqueeze(0)

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

    Q = PARAFAC(dims=nS + nA, k=k, scale=0.1)
    opt = torch.optim.Adamax(Q.parameters(), lr=lr)
    Gs = [[] for _ in range(nT)]
    for episode in range(E):
        for env_idx, env in enumerate(envs):
            s, _ = env.reset()
            s_idx = torch.tensor(discretizer.get_state_index(s)).unsqueeze(0)
            for h in range(H):
                a, a_idx = select_action(Q, s_idx, eps)
                a_idx = torch.tensor(a_idx).unsqueeze(0)
                sp, r, d, _, _ = env.step(a)
                sp_idx = torch.tensor(discretizer.get_state_index(sp)).unsqueeze(0)

                s_idx_train = torch.tensor(discretizer.get_state_index(s)).unsqueeze(0)
                sp_idx_train = torch.tensor(discretizer.get_state_index(sp)).unsqueeze(0)
                update_model(s_idx_train, sp_idx_train, a_idx, r, Q, opt, tasks=None)

                if d:
                    break

                s = sp
                s_idx = sp_idx
                eps = max(eps*eps_decay, eps_min)
            if episode % 10 == 0:
                G = run_test_episode(Q, env_idx)
                Gs[env_idx].append(G)
        print(f"\rEpoch: {episode} - Return: {[Gs[i][-1] for i in range(nT)]}", end="", flush=True)
    return Gs

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(partial(run_experiment, E=E, H=H, lr=lr, eps=eps, eps_decay=eps_decay, eps_min=eps_min, k=k), range(num_experiments))
    
    pool.close()
    pool.join()

    np.save("e3_wire_single_all.npy", results)
