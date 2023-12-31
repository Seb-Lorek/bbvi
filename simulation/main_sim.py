"""
Main simulation script.
"""

import sim_fun.sim_consistency as sim1
import post_process.process_consistency as proc1

import sim_fun.sim_post_dens as sim2
import post_process.process_post_dens as proc2

import sim_fun.sim_runtime as sim3
import post_process.process_runtime as proc3

import subprocess
import os 
import sys
import pickle
import time
import pandas as pd 
import jax

if __name__ == "__main__":
    
    # Create the trace plots 
    # Set the path to execute the script 
    print("Start trace plots")
    script_path = "simulation/sim_fun/sim_trace_elbo.py"

    # Specify the filename for exporting the plots
    current_directory = os.getcwd()
    safe_path = "simulation/plots"

    # Create the full path to the simulation/plots directory
    folder_path = os.path.join(current_directory, safe_path)

    # check if simulation/plots directory already exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    else:
        print(f"Directory '{folder_path}' already exists.")

    # Create the full path to the thesis/assets/plots directory 
    safe_path = "thesis/assets/plots"
    folder_path = os.path.join(current_directory, safe_path)

    # Check if thesis/assets/plots directory already exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    else:
        print(f"Directory '{folder_path}' already exists.")

    # Run the script
    subprocess.run(['python3', script_path])
    
    # Start simulation study 1
    print("Start simulation 1")
    # Track start time simulation 1
    start_time = time.time()

    # Specify the filename for exporting the simulation results 
    safe_path = "simulation/results"

    # Create the full path to the results data directory
    folder_path = os.path.join(current_directory, safe_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    else:
        print(f"Directory '{folder_path}' already exists.")
    
    # Set the number of simulation runs
    n_sim = 200

    # Set the starting key for simulation 1 
    key = jax.random.PRNGKey(2989325200)

    # Run the simulation
    results1, var_grid1 = sim1.sim_fun(n_sim, key)

    # Create the full path to the file
    filename = "sim1_raw.pickle"
    full_filepath = os.path.join(folder_path, filename)

    # Export the object using pickle
    with open(full_filepath, "wb") as file:
        pickle.dump(results1, file)

    filename = "sim1_grid.pickle"
    full_filepath = os.path.join(folder_path, filename)

    # Export the object using pickle
    with open(full_filepath, "wb") as file:
        pickle.dump(var_grid1, file)

    # Check for missing values 
    missing_data, exist_missing = proc1.check_missing_data(results1)

    # Export missing data only if there are missing values 
    if exist_missing: 
        filename = "sim1_missing.pickle"
        full_filepath = os.path.join(folder_path, filename)
        with open(full_filepath, "wb") as file:
            pickle.dump(missing_data, file)

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
    
    filename = "sim1_proc_tiger.pickle"
    full_filepath = os.path.join(folder_path, filename)

    # Export the object using pickle
    with open(full_filepath, "wb") as file:
        pickle.dump(results_proc1_tiger, file)

    filename = "sim1_proc_liesel.pickle"
    full_filepath = os.path.join(folder_path, filename)

    # Export the object using pickle
    with open(full_filepath, "wb") as file:
        pickle.dump(results_proc1_liesel, file)

    # Create the latex table for the results 
    proc1.create_latex_table(results_proc1_tiger, lib="tigerpy")
    proc1.create_latex_table(results_proc1_liesel, lib="liesel")

    # Track end time of simulation 1
    end_time = time.time()
    time_elapsed = end_time - start_time 
    print(f"Time elapsed for simulation 1 n_sim={n_sim}:{time_elapsed/60:.2f} minutes")

    # Start simulation study 2
    print("Start simulation 2")
    
    # Track start time simulation 2
    start_time = time.time() 

    # Set the starting key for simulation 2 
    key = jax.random.PRNGKey(175380738)
    key, subkey = jax.random.split(key)
    n_sim = 100
    
    plot_results2, sim_results2, var_grid2 = sim2.sim_fun(n_sim, subkey)

    # Create the full path to the file
    filename = "sim2_raw.pickle"
    full_filepath = os.path.join(folder_path, filename)

    # Export the object using pickle
    with open(full_filepath, "wb") as file:
        pickle.dump(sim_results2, file)

    # Check for missing values 
    missing_data, exist_missing = proc2.check_missing_data(sim_results2)

    # Export missing data only if there are missing values 
    if exist_missing: 
        filename = "sim2_missing.pickle"
        full_filepath = os.path.join(folder_path, filename)
        with open(full_filepath, "wb") as file:
            pickle.dump(missing_data, file)

    proc2.plot_post_dens(plot_results2)

    sim2_proc = proc2.calc_wasserstein(sim_results2)

    # Create the full path to the file
    filename = "sim2_proc.pickle"
    full_filepath = os.path.join(folder_path, filename)

    # Export the object using pickle
    with open(full_filepath, "wb") as file:
        pickle.dump(sim2_proc, file)

    proc2.plot_wasserstein(sim2_proc)

    proc2.create_latex_table(sim2_proc)

    # Track end time of simulation 2
    end_time = time.time()
    time_elapsed = end_time - start_time 
    print(f"Time elapsed for simulation 2 n_sim={n_sim}:{time_elapsed/60:.2f} minutes")

    # Start simulation study 3 
    print("Start simulation study 3")
    start_time = time.time()

    key = jax.random.PRNGKey(896992974)
    n_sim = 10
    
    results3 = sim3.sim_fun(n_sim, key)

    # Create the full path to the file
    filename = "sim3_raw.pickle"
    full_filepath = os.path.join(folder_path, filename)

    # Export the object using pickle
    with open(full_filepath, "wb") as file:
        pickle.dump(results3, file)

    proc3.create_latex_table(results3)

    # Track end time of simulation 3
    end_time = time.time()
    time_elapsed = end_time - start_time 
    print(f"Time elapsed for simulation 3 n_sim={n_sim}:{time_elapsed/60:.2f} minutes")