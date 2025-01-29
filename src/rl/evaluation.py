import torch


def run_test_episode(Q, env, H, discretizer, env_id=None):
    with torch.no_grad():
        G = 0
        s, _ = env.reset()

        if env_id is not None:
            s_idx = torch.tensor(tuple([env_id]) + discretizer.get_state_index(s)).unsqueeze(0)
        else:
            s_idx = torch.tensor(discretizer.get_state_index(s)).unsqueeze(0)

        for _ in range(H):
            a_idx = Q(s_idx).argmax().item()
            a = discretizer.get_action_from_index(a_idx)
            sp, r, d, _, _ = env.step(a)

            if env_id is not None:
                sp_idx = torch.tensor(tuple([env_id]) + discretizer.get_state_index(sp)).unsqueeze(0)
            else:
                sp_idx = torch.tensor(discretizer.get_state_index(sp)).unsqueeze(0)

            G += r
            if d:
                break

            s = sp
            s_idx = sp_idx

    return G