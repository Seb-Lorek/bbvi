"""
Main simulation script.
"""

import sim_fun.sim_consistency as sim1

if __name__ == "__main__":
    
    # Run the simulation
    results1 = sim1.sim_fun(250)
    
    # Perform analysis on the results
    analysis.analyze_results(results)