"""
This file is ment to reproduce figures for the manuscript “Understanding Electrical Brain Stimulation via the
Reciprocity Theorem” by Torbjørn V. Ness, Christof Koch and Gaute T. Einevoll

"""
import os
from os.path import join
from src import main

root_dir = main.root_dir

fig_folder = join(root_dir, 'figs')
results_folder = join(root_dir, 'sim_results')
os.makedirs(fig_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)


# Figure 2:
main.make_RT_validation_plot(results_folder, fig_folder)

# Figure 3:
main.neural_elements_fig(results_folder, fig_folder)

# Figure 4:
main.analytic_ext_pot(fig_folder)

# Figure 5:
main.detailed_head_model_neuron_fig(results_folder, fig_folder)
