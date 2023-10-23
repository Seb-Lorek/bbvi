"""
Main simulation script.
"""

import sim_fun.sim_consistency as sim1
import post_process.process_consistency as proc1
import subprocess

import os 
import sys
import pickle
import time
import pandas as pd 

if __name__ == "__main__":
    # Track start time 
    start_time = time.time()
    
    # Create the trace plots 
    # Set the path to execute the script 
    print("Trace plots")
    script_path = 'simulation/sim_fun/trace.py'

    # Run the script
    subprocess.run(['python3', script_path])

    # Start simulation study 1
    n_sim = 50

    # Specify the filename for exporting the simulation results 
    current_directory = os.getcwd()
    safe_path = "simulation/results"

    # Create the full path to the results data directory
    folder_path = os.path.join(current_directory, safe_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    else:
        print(f"Directory '{folder_path}' already exists.")
    
    # Run the simulation
    results1, var_grid1 = sim1.sim_fun(n_sim)

    # Create the full path to the file
    filename = "sim1_raw.pickle"
    full_filepath = os.path.join(folder_path, filename)

    # Export the object using pickle
    with open(full_filepath, "wb") as file:
        pickle.dump(results1, file)

    data_plot = proc1.create_data_plot(results1)
    # Plot the univariate distributions 
    proc1.plot_univariate(data_plot)
    
    # Track end time
    end_time = time.time()
    time_elapsed = end_time - start_time 
    print(f"Time elapsed for simulation 1 n_sim={n_sim}:{time_elapsed/60} minutes")

    # Perform analysis on the results
    # analysis1 = proc1.analyze_results(results1)

    # filename = "sim1_proc.pickle"
    # full_filepath = os.path.join(folder_path, filename)

    # Export the object using pickle
    # with open(filename, "wb") as file:
    #    pickle.dump(analysis1, full_filepath)