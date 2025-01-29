import numpy as np
import torch

from src.rl.training import select_action, update_model
from src.rl.evaluation import run_test_episode


class ExperimentRunner:
    def __init__(self, envs, discretizer, model_class, config, experiment_type):
        self.envs = envs
        self.discretizer = discretizer
        self.model_class = model_class
        self.config = config
        self.experiment_type = experiment_type

    def run_experiment(self, exp_num):
        np.random.seed(exp_num)
        torch.manual_seed(exp_num)

        Gs = [[] for _ in range(len(self.envs))]
        if self.experiment_type == "single":
            Qs = [self.model_class(dims=self.discretizer.bucket_states.tolist() + self.discretizer.bucket_actions.tolist(), k=self.config["k"], scale=0.1) for _ in self.envs]
            opts = [torch.optim.Adamax(Q.parameters(), lr=self.config["lr"]) for Q in Qs]
        else:
            dims = self.discretizer.bucket_states.tolist() + self.discretizer.bucket_actions.tolist()
            if self.experiment_type == "multi":
                dims = [len(self.envs)] + dims
            Q = self.model_class(dims=dims, k=self.config["k"], scale=0.1)
            opt = torch.optim.Adamax(Q.parameters(), lr=self.config["lr"])

        for episode in range(self.config["E"]):
            for env_idx, env in enumerate(self.envs):
                s, _ = env.reset()

                if self.experiment_type == "single":
                    Q_model, optimizer = Qs[env_idx], opts[env_idx]
                    s_idx = torch.tensor(self.discretizer.get_state_index(s)).unsqueeze(0)
                elif self.experiment_type == "single_all":
                    Q_model, optimizer = Q, opt
                    s_idx = torch.tensor(self.discretizer.get_state_index(s)).unsqueeze(0)
                else:
                    Q_model, optimizer = Q, opt
                    s_idx = torch.tensor(tuple([env_idx]) + self.discretizer.get_state_index(s)).unsqueeze(0)

                for _ in range(self.config["H"]):
                    a, a_idx = select_action(Q_model, s_idx, self.config["eps"], self.discretizer)
                    a_idx = torch.tensor(a_idx).unsqueeze(0)
                    sp, r, d, _, _ = env.step(a)

                    if self.experiment_type == "single":
                        sp_idx = torch.tensor(self.discretizer.get_state_index(sp)).unsqueeze(0)
                    elif self.experiment_type == "single_all":
                        sp_idx = torch.tensor(self.discretizer.get_state_index(sp)).unsqueeze(0)
                    else:
                        sp_idx = torch.tensor(tuple([env_idx]) + self.discretizer.get_state_index(sp)).unsqueeze(0)

                    update_model(s_idx, sp_idx, a_idx, r, Q_model, optimizer, gamma=self.config["gamma"])

                    if d:
                        break

                    s = sp
                    s_idx = sp_idx
                    self.config["eps"] = max(self.config["eps"] * self.config["eps_decay"], self.config["eps_min"])

                G = run_test_episode(Q_model, env, self.config["H"], self.discretizer, env_id=env_idx if self.experiment_type == "multi" else None)
                Gs[env_idx].append(G)

            print(f"\rEpoch: {episode} - Return: {[Gs[i][-1] for i in range(len(self.envs))]}", end="", flush=True)

        return Gs
