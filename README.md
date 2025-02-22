# A Tensor Low-Rank Approximation for Value Functions in Multi-Task Reinforcement Learning

by
Sergio Rozada,
Santiago Paternain,
Juan Andrés Bazerque,
and Antonio G. Marques,

This code belongs to a paper that has been published in *Asilomar Conference on Signals, Systems, and Computers 2024*.

## TLDR

> We propose a low-rank tensor approach to multi-task reinforcement learning that infers task similarity from data, reducing the need for massive data collection, and demonstrate its efficiency in both benchmark and real-world environments.

<p align="center">
    <img src="figures/fig_1.png" alt="drawing" width="400"/>
</p>


## Abstract

> In pursuit of reinforcement learning systems that could train in physical environments, we investigate multi-task approaches as a means to alleviate the need for massive data acquisition. In a tabular scenario where the Q-functions are collected across tasks, we model our learning problem as optimizing a higher order tensor structure. Recognizing that close-related tasks may require similar actions, our proposed method imposes a low-rank condition on this aggregated Q-tensor. The rationale behind this approach to multi-task learning is that the low-rank structure enforces the notion of similarity, without the need to explicitly prescribe which tasks are similar, but inferring this information from a reduced amount of data simultaneously with the stochastic optimization of the Q-tensor. The efficiency of our low-rank tensor approach to multi-task learning is demonstrated in two numerical experiments, first in a benchmark environment formed by a collection of inverted pendulums, and then into a practical scenario involving multiple wireless communication devices.


## Software implementation

All source code used to generate the results and figures in the paper are in the `src` folder. The calculations and figure generation are all done by running:
* `main.py`: Runs all the experiments and stores the data.
* `plots.ipynb`: Code to replicate the figures of the paper.

Results generated by the code are saved in `results`, and figures are saved in `figures`.

## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://github.com/sergiorozada12/multi-task-lrrl) repository:

    git clone https://github.com/sergiorozada12/multi-task-lrrl.git

or [download a zip archive](https://github.com/sergiorozada12/multi-task-lrrl/archive/refs/heads/main.zip).

## Dependencies

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through [virtual environments](https://docs.python.org/3/library/venv.html). The required dependencies are specified in the file `requirements.txt`.
