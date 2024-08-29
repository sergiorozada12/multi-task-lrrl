import numpy as np
from src.experiments import ExperimentRunner


E = 2
H = 100
nS = 20
nA = 20
gs = [0.5, 1.0, 2.0, 10.0, 100.0]
m = 1
l = 1

MODEL = "single_task" # single_task / single_task_all / multi_task
MODEL_NAME = "results/single_task_1_sample_{E}.npy"
TASK_ID = 1
N_EXPS = 100

k = 100
lr = 0.001
gamma = 0.9
epochs = 1_000


if __name__ == "__main__":
    experiment_runner = ExperimentRunner(gs, m, l, E, H, nS, nA, k, lr, gamma, epochs, N_EXPS)

    res = experiment_runner.run_experiments_distributed(MODEL, task_id=TASK_ID)
    res = np.array(res)

    np.save(MODEL_NAME, res)
