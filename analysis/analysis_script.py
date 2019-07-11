import pandas as pd 
import numpy as np
import os 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# #%% parameters for dev
# experiment_name = "CE_random_patch_2_preprocessed"
# variables_to_show = ["train_loss", "val_loss"]
# n = 1
# 
# =============================================================================
#%% Paths
results_base_dir = os.path.join(os.path.dirname(__file__), os.pardir, "results")


#%%
def show_traces(experiment_name, n, variables_to_show, logy=False, results_base_dir=results_base_dir):
    
    # read summary file
    if n == 1: # only one seed
        summary_path = os.path.join(results_base_dir, experiment_name, "result_outputs", "summary.csv")
        df = pd.read_csv(summary_path, index_col="curr_epoch")
        
    plt.figure()
    for variable in variables_to_show:
        (df.loc[:,variable]).plot(legend=True, logy=logy)
        
# =============================================================================
# #%% dev
# show_traces(experiment_name, n, variables_to_show, results_base_dir)
# =============================================================================
