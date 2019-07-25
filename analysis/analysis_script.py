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

# parameters
min_keywords = ["loss","mse"] # if these keywords appear in the variable name, then less is better
max_keywords = ["auc", "acc"]

#%%
def show_traces_multi_exp(experiment_names, ns, variables_to_show="all", logy=False, results_base_dir=results_base_dir):
    """ Just show the traces of several experiments, so that I don't have to type it all indivdually.
    variables_to_show can be just one list of variables, in which case these variables count for every experiment
    Or it can be a list of list.
    
    
    """
    for experiment_name, n in zip(experiment_names, ns):
        show_traces(experiment_name, n, variables_to_show, logy=logy, results_base_dir=results_base_dir)


def show_traces(experiment_name, n, variables_to_show, logy=False, results_base_dir=results_base_dir):
    """ show traces over the course of training and print peak values """ 

    df = load_summary_file(experiment_name, n, results_base_dir)
    peak_value_df, peak_epoch_df = create_peak_value_df(df)

    if variables_to_show == "all":
        variables_to_show = list(df.columns)


    plt.figure()
    for variable in variables_to_show:
        (df.loc[:,variable]).plot(legend=True, logy=logy, title=experiment_name)
        # print peak stats
        peak_value = peak_value_df.loc[0,variable]
        peak_epoch = peak_epoch_df.loc[0,variable]

        print("peak " + variable + ": {:.4f}".format(peak_value) + " in epoch: " + str(peak_epoch))

    display(plt.gcf())
    plt.close()

def print_table_peak_values(experiment_names, ns, variables_to_show="all", results_base_dir=results_base_dir):    
    for experiment_name,n in zip(experiment_names, ns):            
        df = load_summary_file(experiment_name, n, results_base_dir)
        peak_value_df, peak_epoch_df = create_peak_value_df(df)
        peak_value_df.insert(0, "experiment", experiment_name)

#        print(peak_value_df)
        try: # create peak_values if it doesn't exist yet
            peak_values_df
        except:
            peak_values_df = pd.DataFrame(columns= ["experiment"] + list(df.columns))
        
#        print(peak_values_df)
        peak_values_df = peak_values_df.append(peak_value_df)
#        print(peak_values_df)

    if variables_to_show == "all":
        variables_to_show = list(peak_values_df.columns)
    else:
        variables_to_show = ["experiment"] + variables_to_show
    
    display(peak_values_df.loc[:,variables_to_show])
    
    
def create_peak_value_df(df,min_keywords=min_keywords, max_keywords=max_keywords):
    peak_value_df = pd.DataFrame(columns=list(df.columns))
    peak_epoch_df = pd.DataFrame(columns=list(df.columns)) # epochs where these peak values occur
    for variable in list(df.columns):
        peak_at_min = any([keyword in variable for keyword in min_keywords])
        peak_at_max = any([keyword in variable for keyword in max_keywords])
        if peak_at_min:
            peak_value = df.loc[:,variable].min(axis=0)
            peak_epoch = df.loc[:,variable].idxmin(axis=0)        
        if peak_at_max:
            peak_value = df.loc[:,variable].max(axis=0)
            peak_epoch = df.loc[:,variable].idxmax(axis=0)
        peak_value_df.loc[0,variable] = peak_value
        peak_epoch_df.loc[0,variable] = peak_epoch
    
    return peak_value_df, peak_epoch_df

    
def load_summary_file(experiment_name, n, results_base_dir):
    # read summary file
    if n == 1: # only one seed
        summary_path = os.path.join(results_base_dir, experiment_name, "result_outputs", "summary.csv")
        df = pd.read_csv(summary_path, index_col="curr_epoch")
    
    return df


# =============================================================================
# #%% dev
# show_traces(experiment_name, n, variables_to_show, results_base_dir)
# =============================================================================
