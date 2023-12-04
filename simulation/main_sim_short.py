"""
Main simulation script.
"""

import sim_fun.sim_consistency as sim1
import post_process.process_consistency as proc1

import sim_fun.sim_post_dens as sim2
import post_process.process_post_dens as proc2

import subprocess
import os 
import sys
import pickle
import time
import pandas as pd 
import jax

if __name__ == "__main__":
    
    # Start processing of simulation results 
    print("Start short simulation with results.")
    
    # Track start time of processing simulation results 
    start_time = time.time() 

    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, "simulation/results")

    # Import simulation results 1 
    filename = "sim1_raw.pickle"
    full_filepath = os.path.join(folder_path, filename)

    with open(full_filepath, 'rb') as file:
        results1 = pickle.load(file)

    filename = "sim1_grid.pickle"
    full_filepath = os.path.join(folder_path, filename) 
    
    with open(full_filepath, 'rb') as file:
        var_grid1 = pickle.load(file)

    # Split the data set by the different methods
    results_tiger, results_liesel = proc1.split_libs(results1)

    # Plot the univariate distributions for tiger 
    data_plot_tiger = proc1.create_data_plot(results_tiger, var_grid1)
    proc1.plot_univariate(data_plot_tiger, lib="tigerpy")

    # Plot the univariate distributions for tiger 
    data_plot_liesel = proc1.create_data_plot(results_liesel, var_grid1)
    proc1.plot_univariate(data_plot_liesel, lib="liesel")

    # Create the performance measure dataset for both methods
    results_proc1_tiger = proc1.analyze_results(results_tiger, var_grid1)
    results_proc1_liesel = proc1.analyze_results(results_liesel, var_grid1)

    # Create the latex table for the results 
    proc1.create_latex_table(results_proc1_tiger, lib="tigerpy")
    proc1.create_latex_table(results_proc1_liesel, lib="liesel")

    # Import simulation results 2
    filename = "sim2_raw.pickle"
    full_filepath = os.path.join(folder_path, filename)

    with open(full_filepath, 'rb') as file:
        sim_results2 = pickle.load(file)

    sim2_proc = proc2.calc_wasserstein(sim_results2)

    proc2.plot_wasserstein(sim2_proc)

    proc2.create_latex_table(sim2_proc)

    # Track end time of processing simulation results 
    end_time = time.time()
    time_elapsed = end_time - start_time 
    print(f"Time elapsed for processing the simulation results:{time_elapsed/60:.2f} minutes")