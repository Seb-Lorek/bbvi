"""
Main simulation script.
"""

import sim_fun.sim_consistency as sim1
import post_process.process_consistency as proc1

import os 
import sys
import pickle
import time

if __name__ == "__main__":
    # Track start time 
    start_time = time.time()
    n = 20

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
    results1 = sim1.sim_fun(n)
    
    # Create the full path to the file
    filename = "sim1_raw.pickle"
    full_filepath = os.path.join(folder_path, filename)

    # Export the object using pickle
    with open(full_filepath, "wb") as file:
        pickle.dump(results1, file)
    
    # Track end time
    end_time = time.time()
    time_elapsed = end_time - start_time 
    print(f"Time elapsed for n_sim={n}:{time_elapsed/60} minutes")

    # Perform analysis on the results
    # analysis1 = proc1.analyze_results(results1)

    # filename = "sim1_proc.pickle"
    # full_filepath = os.path.join(folder_path, filename)

    # Export the object using pickle
    # with open(filename, "wb") as file:
    #    pickle.dump(analysis1, full_filepath)