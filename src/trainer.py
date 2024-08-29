import torch

from src.environments import PendulumEnv
from src.utils import Discretizer


class QNetworkTrainer:
    def __init__(self, Q, lr, gamma=0.9):
        self.Q = Q
        self.gamma = gamma
        self.opt = torch.optim.Adamax(Q.parameters(), lr=lr)

    def unpack_batch(self, batch, use_tasks=False):
        if len(batch) == 4:
            if use_tasks:
                raise ValueError("use_tasks is True but batch doesn't contain tasks.")
            tasks = None
            states, states_next, actions, rewards = batch
        elif len(batch) == 5:
            if use_tasks:
                tasks, states, states_next, actions, rewards = batch
            else:
                tasks = None
                _, states, states_next, actions, rewards = batch
        else:
            raise ValueError("Unexpected batch length. Expected 4 or 5 elements per batch.")
        
        return tasks, states, states_next, actions, rewards

    def create_target(self, states_next, rewards, tasks=None):
        if tasks is not None:
            idx_target = torch.cat((tasks.unsqueeze(1), states_next), dim=1)
        else:
            idx_target = states_next

        with torch.no_grad():
            q_target = rewards + self.gamma * self.Q(idx_target).max(dim=1).values

        return q_target

    def create_idx_hat(self, states, actions, tasks=None):
        if tasks is not None:
            idx_hat = torch.cat((tasks.unsqueeze(1), states, actions.unsqueeze(1)), dim=1)
        else:
            idx_hat = torch.cat((states, actions.unsqueeze(1)), dim=1)
        
        return idx_hat

    def train(self, loader, epochs, use_tasks=False):
        for e in range(epochs):
            total_loss = 0
            for i, batch in enumerate(loader):
                tasks, states, states_next, actions, rewards = self.unpack_batch(batch, use_tasks)

                for factor in self.Q.factors:
                    q_target = self.create_target(states_next, rewards, tasks)
                    idx_hat = self.create_idx_hat(states, actions, tasks)
                    q_hat = self.Q(idx_hat)

                    self.opt.zero_grad()
                    loss = torch.nn.MSELoss()(q_hat, q_target)
                    loss.backward()

                    # Zero the gradients of the other factors
                    with torch.no_grad():
                        for frozen_factor in self.Q.factors:
                            if frozen_factor is not factor:
                                frozen_factor.grad = None

                    self.opt.step()

                total_loss += loss.item()

            print(f"\rEpoch: {e} - Loss: {total_loss / (i + 1)}", end="")


class QNetworkTester:
    def __init__(self, Q, nS, nA, gs=[1], m=1, l=1, H=100):
        self.Q = Q
        self.nS = nS
        self.nA = nA
        self.gs = gs
        self.m = m
        self.l = l
        self.H = H

        self.envs = [PendulumEnv(g, m, l) for g in gs]
        self.discretizer = Discretizer(
            min_points_states=[-1, -5],
            max_points_states=[1, 5],
            bucket_states=[nS] * 2,
            min_points_actions=[-2],
            max_points_actions=[2],
            bucket_actions=[nA],
        )

    def run_episode(self, env, task_id=None):
        G = 0
        s, _ = env.reset()
        s_idx = self.get_state_index(s, task_id)

        for _ in range(self.H):
            s_ten = torch.tensor(s_idx).unsqueeze(0)
            a_idx = self.Q(s_ten).argmax().item()
            a = self.discretizer.get_action_from_index(a_idx)
            s, r, d, _, _ = env.step(a)
            s_idx = self.get_state_index(s, task_id)

            G += r

            if d:
                break

        return G

    def get_state_index(self, state, task_id=None):
        s_idx = self.discretizer.get_state_index(state)
        if task_id is not None:
            return tuple([task_id] + list(s_idx))
        return tuple(s_idx)

    def test(self, task_id=None, multi_task=False):
        Gs = []
        if task_id is not None:
            env = self.envs[task_id]
            G = self.run_episode(env, task_id)
            Gs.append(G)
        elif multi_task:
            for i, env in enumerate(self.envs):
                G = self.run_episode(env, task_id=i)
                Gs.append(G)
        else:
            for i, env in enumerate(self.envs):
                env = self.envs[i]
                G = self.run_episode(env, i)
                Gs.append(G)

        return Gs
