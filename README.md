# network-reconstruction

This is the code for my Bachelor's thesis on network reconstruction using graph neural networks.
The thesis is titled "Learning Dynamical Processes to Infer the Underlying Network Structure" and is also in this repository as a PDF.

Requires Python 3.7.9 and PyTorch 1.1.0.

The folder "notebooks" contains the Jupyter Notebooks for data generation and different visualization scripts as well as the code for the correlation-based baselines.
The code for hill climbing and simulated annealing can be found in the "Hillclimbing" folder (files: hillclimbing_linear.py and simulated_annealing.py) and the code for PBLS can be found in the main_nodewise.py file in the "PBLS" folder.
The "common" folder contains utilities that are shared between all methods.
The folder "SigmoidGraphNetworks" contains our adapted implementation of Zhang et al.'s GGN that we use as a baseline. When the flag USE_GUMBEL in the file train_network is set to False, another approach is used which replaces the Gumbel Sampling procedure with a sigmoid function (this was not discussed in the thesis).