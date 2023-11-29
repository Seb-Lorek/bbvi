"""
Process the results from the simulation to obtain the targets.
"""

import sys
import os 

import numpy as np
import pandas as pd
import jax 
import jax.numpy as jnp

import matplotlib.pyplot as plt
import seaborn as sns

def check_missing_data(results):
    missing_data = []
    for j, result in enumerate(results):
        data_dict = {}
        keys = [key for key in result.keys() if "_true" not in key]
        for key in keys:
            if np.any(np.isnan(result[key])):
                data_dict[key] = np.argwhere(np.isnan(result[key]))
            else:
                data_dict[key] = None
        missing_data.append(data_dict)
    
    exist_missing = False
    for data_dict in missing_data:
        for value in data_dict.values():
            if value != None:
                exist_missing = True
                break
        if exist_missing:
            break

    if exist_missing:
        print("Consistency simulation (sim1) contains missing values.")

    return missing_data, exist_missing

def create_data_plot(results, grid):
    plot_data = {}
    store = []
    for j, result in enumerate(results): 
        data_dict = {}
        keys = [key for key in result.keys() if "_true" not in key]
        for key in keys:
            data = pd.DataFrame(result[key], columns=[key + "_" + str(i) for i in range(result[key].shape[-1])])
            appended_data = pd.DataFrame({"n_obs":[grid[j]["n_obs"]]*grid[j]["n_sim"]})
            data = pd.concat([data, appended_data], axis=1)
            data_dict[key] = data
        store.append(data_dict)
    for j, result in enumerate(store):
        for key, value in result.items():
            if grid[j]["response_dist"] in plot_data.keys() and key in plot_data[grid[j]["response_dist"]].keys():
                plot_data[grid[j]["response_dist"]][key] = pd.concat([plot_data[grid[j]["response_dist"]][key], value], ignore_index=True)
            elif grid[j]["response_dist"] in plot_data.keys() and key not in plot_data[grid[j]["response_dist"]].keys():
                plot_data[grid[j]["response_dist"]][key] = value
            else:
                plot_data[grid[j]["response_dist"]] = {}
                plot_data[grid[j]["response_dist"]][key] = value

    return plot_data

def plot_univariate(data_plot):

    current_path = os.getcwd()
    folder_path = os.path.join(current_path, "simulation/plots/consistency")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    else:
        print(f"Directory '{folder_path}' already exists.")

    true_coefs = {"normal": {"loc": [3.0, 0.2, -0.5], "scale":[1.0]}, 
                  "bernoulli": {"logit": [1.0, 2.0]}}

    fig_count = 1
    for response_dist, true_coef in zip(data_plot.values(), true_coefs.values()):
        for est, true in zip(response_dist.items(), true_coef.items()):
            key, value = est
            key_true, value_true = true
            filename = "plot" + "_" + str(fig_count) + ".pdf"
            full_filepath = os.path.join(folder_path, 
                                         filename)
            sns.set_style(style='whitegrid')
            ncols = value.shape[1] - 1
            if ncols == 1:
                param = key + "_" + str(0)
                if key == "scale":
                    label = r"\hat{\sigma}"
                fig, ax = plt.subplots(figsize=(10,6))
                ax.axvline(x=value_true, color="grey", linestyle= "--", alpha=0.6)
                sns.kdeplot(value, x=param, hue="n_obs", bw_adjust=1.5, palette="colorblind")
                ax.set_xlabel(rf"${label}$")
            else: 
                fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, 6))
                for i in range(ncols):
                    param = key + "_" + str(i)
                    if key in ["loc", "logits"]:
                        label = r"\hat{\beta}"
                        axs[i].axvline(x=value_true[i], color="grey", linestyle= "--", alpha=0.6)
                    sns.kdeplot(value, x=param, hue="n_obs", bw_adjust=1.5, ax=axs[i], palette="colorblind")
                    axs[i].set_xlabel(rf"${label}_{{{str(i)}}}$")
                    axs[i].set_ylabel("")
                fig.supylabel("Density")
            plt.tight_layout()
            plt.savefig(full_filepath)
            sns.reset_orig()
            fig_count += 1

def analyze_results(results, grid):
    results_cons = {}
    for j, result in enumerate(results):

        keys_est = [key for key in result.keys() if "_true" not in key]
        keys_true = [key for key in result.keys() if "_true" in key]

        for key_est, key_true in zip(keys_est, keys_true):
            # Calculate the performance measures 
            bias = calc_bias(result[key_est], result[key_true])
            mc_se_bias = calc_mc_se_bias(result[key_est])
            emp_se = calc_empse(result[key_est])
            mc_se_emp_se = calc_mc_se_empse(result[key_est])
            data = pd.DataFrame({"Bias": bias, 
                                 "MC_SE_Bias": mc_se_bias, 
                                 "EmpSE": emp_se,
                                 "MC_SE_EmpSE": mc_se_emp_se})
            appended_data = pd.DataFrame({"n_obs":[grid[j]["n_obs"]]*bias.shape[0]})
            data = pd.concat([data, appended_data], axis=1)
            data.index = [key_est + "_" + str(i) for i in range(bias.shape[0])]
            if grid[j]["response_dist"] in results_cons.keys() and key_est in results_cons[grid[j]["response_dist"]].keys():
                results_cons[grid[j]["response_dist"]][key_est] = pd.concat([results_cons[grid[j]["response_dist"]][key_est], data])
            elif grid[j]["response_dist"] in results_cons.keys() and key_est not in results_cons[grid[j]["response_dist"]].keys():
                results_cons[grid[j]["response_dist"]][key_est] = data
            else:
                results_cons[grid[j]["response_dist"]] = {}
                results_cons[grid[j]["response_dist"]][key_est] = data

    return results_cons

# Calculate bias 
def calc_bias(result, true):
    diff = result - true 
    bias = np.mean(diff, axis=0)
    return bias 

# Calculate Monte Carlo SE of bias
def calc_mc_se_bias(result):
    mean = np.mean(result, axis=0)
    mc_var_bias = 1/(result.shape[0]*(result.shape[0] - 1)) * np.sum((result - mean)**2, axis=0)
    return np.sqrt(mc_var_bias)

# Calculate EmpSE 
def calc_empse(result):
    mean = np.mean(result, axis=0)
    emp_var = 1/(result.shape[0] - 1) * np.sum((result - mean)**2, axis=0)
    return np.sqrt(emp_var)

# Calculate Monte Carlo SE of EmpSE
def calc_mc_se_empse(result):
    mean = np.mean(result, axis=0)
    emp_var = 1/(result.shape[0] - 1) * np.sum((result - mean)**2, axis=0)
    emp_se = np.sqrt(emp_var)
    mc_se_empse = emp_se/np.sqrt(2*(result.shape[0] - 1))
    return mc_se_empse

def create_latex_table(results_proc):
    current_path = os.getcwd()
    folder_path = os.path.join(current_path, "simulation/tables")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    else:
        print(f"Directory '{folder_path}' already exists.")

    table_count = 1
    for result in results_proc.values():
        combine_data = [data.reset_index() for data in result.values()]
        table_data = pd.concat(combine_data, axis=0)
        print_table = table_data.pivot(index="index", columns="n_obs")
        filename = "sim1_table" + "_" + str(table_count) + ".latex"
        full_filepath = os.path.join(folder_path, 
                                     filename)
        print_table.to_latex(buf=full_filepath,
                             index=True,
                             index_names=False,
                             float_format="%.4f")
        table_count += 1
    