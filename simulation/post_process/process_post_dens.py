"""
Process the results from the posterior density simulation to obtain the targets.
"""

import jax
import jax.numpy as jnp

import numpy as np
import pandas as pd
import os 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import (
    norm
)
import ot

def check_missing_data(results):

    def find_missing(x):
        if np.any(np.isnan(x)):
             y = np.argwhere(np.isnan(x))
        else:
            y = None
        return y
    
    missing_data = jax.tree_map(lambda x: find_missing(x), results)
    
    exist_missing = False
    for data_dict in missing_data.values():
        for value in data_dict.values():
            if value != None:
                exist_missing = True
                break
        if exist_missing:
            break

    if exist_missing:
        print("Posterior density simulation 1 (sim2) contains missing values.")

    return missing_data, exist_missing

def plot_post_dens(results: dict) -> None:

    current_path = os.getcwd()
    folder_path = os.path.join(current_path, "simulation/plots/post_density")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    else:
        print(f"Directory '{folder_path}' already exists.")

    keys_tiger = list(results["tigerpy"].keys())

    fig_count = 1
    for kw in keys_tiger:
        arr_tiger = results["tigerpy"][kw]
        arr_liesel = results["liesel"][kw]
        col_tiger = ["BBVI_" + str(i) for i in range(arr_tiger.shape[0])]
        col_liesel = ["MCMC_" + str(i) for i in range(arr_liesel.shape[0])]
        if kw == "beta":
            df_tiger = pd.DataFrame(jnp.squeeze(arr_tiger).T, columns=col_tiger)
            df_liesel = pd.DataFrame(jnp.squeeze(arr_liesel).T, columns=col_liesel)

            filename = "plot" + "_" + str(fig_count) + ".pdf"
            full_filepath = os.path.join(folder_path, 
                                         filename)
    
            sns.set_theme(style="whitegrid")
            fig = plt.figure(figsize=(8,6))

            sns.kdeplot(df_tiger, bw_adjust=1.5, palette="Reds", legend=False)
            sns.kdeplot(df_liesel, bw_adjust=1.5, palette="Blues", legend=False)
            red_patch = mpatches.Patch(color=sns.color_palette("Reds")[4], label='BBVI')
            blue_patch = mpatches.Patch(color=sns.color_palette("Blues")[4], label='MCMC')
            plt.legend(handles=[red_patch, blue_patch])
            plt.xlabel(r"$\beta_{0}$")
            plt.title("Fixed parameter")
            plt.savefig(full_filepath)
            sns.reset_orig()
            fig_count += 1
        elif kw == "gamma":
            params_pos = [0, 6, 13, 18]
            df_tiger = pd.DataFrame(data=jnp.vstack([arr_tiger[:,:,i].T for i in params_pos]), 
                                    columns=col_tiger)
            df_tiger["gamma_pos"] = jnp.repeat(jnp.asarray(params_pos), arr_tiger.shape[1])
            df_liesel = pd.DataFrame(data=jnp.vstack([arr_liesel[:,:,i].T for i in params_pos]), 
                                    columns=col_liesel)
            df_liesel["gamma_pos"] = jnp.repeat(jnp.asarray(params_pos), arr_liesel.shape[1])

            filename = "plot" + "_" + str(fig_count) + ".pdf"
            full_filepath = os.path.join(folder_path, 
                                         filename)
            sns.set_theme(style="whitegrid")
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
            axs = axs.flatten()
            for j, i in enumerate(params_pos):
                df_tiger_sub = df_tiger[df_tiger["gamma_pos"]==i].drop(columns="gamma_pos")
                df_liesel_sub = df_liesel[df_liesel["gamma_pos"]==i].drop(columns="gamma_pos")
                    
                label = r"\tilde{\gamma}"
                sns.kdeplot(df_tiger_sub, bw_adjust=1.5, ax=axs[j], palette="Reds", legend=False)
                sns.kdeplot(df_liesel_sub, bw_adjust=1.5, ax=axs[j], palette="Blues", legend=False)
                axs[j].set_xlabel(rf"${label}_{{{str(i)}}}$")
                axs[j].set_ylabel("")

            red_patch = mpatches.Patch(color=sns.color_palette("Reds")[4], label='BBVI')
            blue_patch = mpatches.Patch(color=sns.color_palette("Blues")[4], label='MCMC')    
            fig.legend(handles=[red_patch, blue_patch], loc="outside upper right")
            fig.supylabel("Density")
            fig.suptitle("Selected smooth parameters")
            plt.tight_layout()
            plt.savefig(full_filepath)
            sns.reset_orig()
            fig_count += 1
        elif kw == "tau2":
            df_tiger = pd.DataFrame(jnp.squeeze(arr_tiger).T, columns=col_tiger)
            df_liesel = pd.DataFrame(jnp.squeeze(arr_liesel).T, columns=col_liesel)

            filename = "plot" + "_" + str(fig_count) + ".pdf"
            full_filepath = os.path.join(folder_path, 
                                         filename)
    
            sns.set_theme(style="whitegrid")
            fig = plt.figure(figsize=(8,6))

            sns.kdeplot(df_tiger, bw_adjust=1.5, palette="Reds", legend=False)
            sns.kdeplot(df_liesel, bw_adjust=1.5, palette="Blues", legend=False)
            red_patch = mpatches.Patch(color=sns.color_palette("Reds")[4], label='BBVI')
            blue_patch = mpatches.Patch(color=sns.color_palette("Blues")[4], label='MCMC')
            plt.legend(handles=[red_patch, blue_patch])
            plt.xlabel(r"$\tau^{2}$")
            plt.title("Inverse smoothing parameter")
            plt.savefig(full_filepath)
            sns.reset_orig()
            fig_count += 1
        elif kw == "sigma":
            df_tiger = pd.DataFrame(jnp.squeeze(arr_tiger).T, columns=col_tiger)
            df_liesel = pd.DataFrame(jnp.squeeze(arr_liesel).T, columns=col_liesel)

            filename = "plot" + "_" + str(fig_count) + ".pdf"
            full_filepath = os.path.join(folder_path, 
                                         filename)
    
            sns.set_theme(style="whitegrid")
            fig = plt.figure(figsize=(8,6))

            sns.kdeplot(df_tiger, bw_adjust=1.5, palette="Reds", legend=False)
            sns.kdeplot(df_liesel, bw_adjust=1.5, palette="Blues", legend=False)
            red_patch = mpatches.Patch(color=sns.color_palette("Reds")[4], label='BBVI')
            blue_patch = mpatches.Patch(color=sns.color_palette("Blues")[4], label='MCMC')
            plt.legend(handles=[red_patch, blue_patch])
            plt.xlabel(r"$\sigma$")
            plt.title("Scale")
            plt.savefig(full_filepath)
            sns.reset_orig()
            fig_count += 1

def calc_wasserstein(results: dict) -> dict:

    kws_tiger = list(results["tigerpy"].keys())
    wasserstein_dist = {}
    for kw in kws_tiger:
        arr_tiger = results["tigerpy"][kw]
        arr_liesel = results["liesel"][kw]

        arr_w = jnp.array([])
        n = arr_tiger.shape[1]
        a, b = jnp.ones((n,)) / n, jnp.ones((n,)) / n  

        for j in range(arr_tiger.shape[0]):
            M = ot.dist(arr_liesel[j,:,:], arr_tiger[j,:,:])
            w = jnp.sqrt(ot.emd2(a, b, M))
            arr_w = jnp.append(arr_w, w)

        wasserstein_dist[kw] = arr_w

    return wasserstein_dist

def plot_wasserstein(results: dict) -> None:

    current_path = os.getcwd()
    folder_path = os.path.join(current_path, "simulation/plots/post_density")

    df = pd.DataFrame(results)
    df.rename({"beta": r"$\beta_{0}$", 
               "gamma": r"$\mathbf{\gamma}$", 
               "sigma": r"$\sigma$", 
               "tau2": r"$\tau^{2}$"}, axis=1, inplace=True)
    col = df.columns
    long_df = pd.melt(df, value_vars=col, var_name="param", value_name="wasserstein")
    long_dfs = []
    for i in col:
        mask = long_df["param"].isin([i])
        long_dfs.append(long_df[mask])

    filename = "plot_box_wasser.pdf"
    full_filepath = os.path.join(folder_path, 
                                 filename)

    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,6))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        sns.stripplot(long_dfs[i], 
                      x="param", 
                      y="wasserstein", 
                      alpha = 0.4, 
                      ax=ax, 
                      color=sns.color_palette("colorblind")[i])
        sns.boxplot(long_dfs[i], 
                    x="param", 
                    y="wasserstein", 
                    ax=ax, 
                    fill=False, 
                    color=sns.color_palette("colorblind")[i])
        ax.set_ylabel("")
        ax.set_xlabel("")

    fig.supylabel(r"$W_{2}$")
    fig.suptitle("Wasserstein distance $W_{2}$ for each parameter block")
    plt.tight_layout()
    plt.savefig(full_filepath)
    sns.reset_orig()
    
def create_latex_table(results: dict) -> None:
    current_path = os.getcwd()
    folder_path = os.path.join(current_path, "simulation/tables")
    results = {kw: np.array(value) for kw, value in results.items()}
    targets = {}
    for kw, value in results.items():
        targets[kw] = calc_performance_measures(value)

    measures = ["mean", "median", "q_25", "q_75", "mean_se", "median_se", "q_25_se", "q_75_se"]
    df = pd.DataFrame(targets, index=measures)

    filename = "sim2_table_1.latex"
    full_filepath = os.path.join(folder_path, 
                                 filename)
    df.to_latex(buf=full_filepath,
                index=True,
                index_names=False,
                float_format="%.4f")

def calc_performance_measures(array: np.ndarray) -> list:
    n_sim = array.shape[0]
    mean = np.mean(array)
    med = np.median(array)
    q_25 = np.quantile(array, q=0.25)
    q_75 = np.quantile(array, q=0.75)

    mean_se = calc_mean_se(array, n_sim)
    med_se = calc_med_se(array, n_sim)
    q_25_se = calc_q_se(array, p=0.25, n_sim=n_sim)
    q_75_se = calc_q_se(array, p=0.75, n_sim=n_sim)

    return [mean, med, q_25, q_75, mean_se, med_se, q_25_se, q_75_se]

def calc_mean_se(array: np.ndarray, n_sim: int):
    return np.std(array)/np.sqrt(n_sim)

def calc_med_se(array: np.ndarray, n_sim: int):
    mean_se = calc_mean_se(array, n_sim)
    return np.sqrt(np.pi/2)*mean_se

def calc_q_se(array: np.ndarray, p: float, n_sim: int):
    num = np.std(array)*np.sqrt(p*(1-p))
    phi = norm.pdf(norm.ppf(p))
    denom = np.sqrt(n_sim)*phi

    return num/denom