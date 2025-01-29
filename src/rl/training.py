import torch
import numpy as np


def create_target(idx_target, rewards, Q, gamma):
    with torch.no_grad():
        q_target = rewards + gamma * Q(idx_target).max(dim=1).values
    return q_target

def create_idx_hat(states, actions):
    return torch.cat((states, actions), dim=1)

def update_model(s_idx, sp_idx, a_idx, r, Q, opt, gamma):
    for factor in Q.factors:
        q_target = create_target(sp_idx, r, Q, gamma)
        idx_hat = create_idx_hat(s_idx, a_idx)
        q_hat = Q(idx_hat)

        opt.zero_grad()
        loss = torch.nn.MSELoss()(q_hat, q_target)
        loss.backward()

        with torch.no_grad():
            for frozen_factor in Q.factors:
                if frozen_factor is not factor:
                    frozen_factor.grad = None

        opt.step()

def select_action(Q, s_idx, epsilon, discretizer):
    if np.random.rand() < epsilon:
        action_idx = np.random.choice(discretizer.bucket_actions[0])
    else:
        with torch.no_grad():
            action_idx = Q(s_idx).argmax().item()

    return discretizer.get_action_from_index(action_idx), torch.tensor(action_idx).unsqueeze(0)
