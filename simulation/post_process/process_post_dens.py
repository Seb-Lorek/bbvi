"""
Process the results from the simulation to obtain the targets.
"""

import jax
import jax.numpy as jnp

import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def plot_post_dens(results: dict):

    current_path = os.getcwd()
    folder_path = os.path.join(current_path, "simulation/plots/post_density")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    else:
        print(f"Directory '{folder_path}' already exists.")

    keys_tiger = list(results["tigerpy"])

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

            sns.kdeplot(df_tiger, palette="Reds", legend=False)
            sns.kdeplot(df_liesel, palette="Blues", legend=False)
            red_patch = mpatches.Patch(color=sns.color_palette("Reds")[4], label='BBVI')
            blue_patch = mpatches.Patch(color=sns.color_palette("Blues")[4], label='MCMC')
            plt.legend(handles=[red_patch, blue_patch])
            plt.xlabel(r"$\beta_{0}$")
            plt.title("Fixed parameter")
            plt.savefig(full_filepath)
            sns.reset_orig()
            fig_count += 1
        elif kw == "gamma":
            params_pos = [0, 6, 13, 19]
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
                count = str(i)
                sns.kdeplot(df_tiger_sub, ax=axs[j], palette="Reds", legend=False)
                sns.kdeplot(df_liesel_sub, ax=axs[j], palette="Blues", legend=False)
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

            sns.kdeplot(df_tiger, palette="Reds", legend=False)
            sns.kdeplot(df_liesel, palette="Blues", legend=False)
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

            sns.kdeplot(df_tiger, palette="Reds", legend=False)
            sns.kdeplot(df_liesel, palette="Blues", legend=False)
            red_patch = mpatches.Patch(color=sns.color_palette("Reds")[4], label='BBVI')
            blue_patch = mpatches.Patch(color=sns.color_palette("Blues")[4], label='MCMC')
            plt.legend(handles=[red_patch, blue_patch])
            plt.xlabel(r"$\sigma$")
            plt.title("Scale")
            plt.savefig(full_filepath)
            sns.reset_orig()
            fig_count += 1