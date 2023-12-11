"""
Process the runtime study.
"""

import jax
import jax.numpy as jnp

import numpy as np
import pandas as pd
import os 

def create_latex_table(results: dict) -> None:
    current_path = os.getcwd()
    folder_path = os.path.join(current_path, "simulation/tables")
    results = {kw: np.array(value) for kw, value in results.items()}
    targets = {}
    for kw, value in results.items():
        targets[kw] = calc_performance_measures(value)

    measures = ["mean", "mean_se"]
    df = pd.DataFrame(targets, index=measures)

    filename = "sim3_table_1.latex"
    full_filepath = os.path.join(folder_path, 
                                 filename)
    df.to_latex(buf=full_filepath,
                index=True,
                index_names=False,
                float_format="%.4f")

def calc_performance_measures(array: np.ndarray) -> list:
    n_sim = array.shape[0]
    mean = np.mean(array)

    mean_se = calc_mean_se(array, n_sim)

    return [mean, mean_se]

def calc_mean_se(array: np.ndarray, n_sim: int):
    return np.std(array)/np.sqrt(n_sim)