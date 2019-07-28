import os

import pandas as pd
import matplotlib.pyplot as plt

# parameters

# =============================================================================
# # for dev
# experiment_names = ['CE_DTD_r2_stand_scale_1___AD_window_mean',
# 'CE_DTD_r2_stand_scale_1___AD_window_min']
# 
# 
# =============================================================================


# paths
results_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "results", "anomaly_detection"))

def summary_stats(experiment_names, sort_column=None, display_table=True):
    """
    Create and print DataFrame that contains the stats (mean + SD) of several measures from several experiments.
    Inputs csv is/are expected to contain columns with column names + values
    """
    summary_df = pd.DataFrame(columns=["experiment_name", "val_mean_aucroc", "val_sd", "val_sem", "test_mean_aucroc", "test_sd", "test_sem"])
    
    for experiment_name in experiment_names:
        curr_exp_stats = {"experiment_name": experiment_name} # stats dicts
        
        for which_set in ["val", "test"]:
            file_name = which_set + "_summary.csv"
            summary_path = os.path.join(results_base_dir, experiment_name, "tables", file_name)
            df = pd.read_csv(summary_path)
            mn = df.loc[:,"aucroc"].mean()
            sd = df.loc[:,"aucroc"].std()
            sem = df.loc[:,"aucroc"].sem()
            if which_set == "val":
                curr_exp_stats["val_mean_aucroc"] = mn
                curr_exp_stats["val_sd"] = sd
                curr_exp_stats["val_sem"] = sem
            elif which_set == "test":
                curr_exp_stats["test_mean_aucroc"] = mn
                curr_exp_stats["test_sd"] = sd
                curr_exp_stats["test_sem"] = sem
        
        summary_df = summary_df.append(curr_exp_stats, ignore_index=True)    
    
    summary_df = summary_df.set_index("experiment_name")
    if sort_column is not None:
        summary_df.sort_values(by=sort_column)
    if display_table:
        display(summary_df)
    return summary_df


def summary_stats_by_train_exp(experiment_names, delimiter="___", display_table=True):
    """
    Summary table over multiple experiments, but all AD experiments that go back to the same train experiment are in one row.
    """
    summary_df = summary_stats(experiment_names, display_table=False)
    train_experiment_names = get_train_experiment_names(summary_df.index, delimiter=delimiter)
    summary_by_train_exp_df = pd.DataFrame(columns=["train_experiment_name"])
    summary_by_train_exp_df.set_index("train_experiment_name")
    for train_experiment_name in train_experiment_names:
        dict_to_append = {"train_experiment_name":train_experiment_name}
        for row in summary_df.itertuples(): # note that row is returned as a namedtuple
            if train_experiment_name in row.Index:
                AD_experiment_name = row.Index.split(delimiter)[1]
                dict_to_append.update({(AD_experiment_name +"_" + k): v for k,v in row._asdict().items() if not "Index" in k})

        summary_by_train_exp_df = summary_by_train_exp_df.append(dict_to_append, ignore_index=True)
    
    summary_by_train_exp_df = summary_by_train_exp_df.set_index("train_experiment_name")
    
    if display_table:
        display(summary_by_train_exp_df)
    return summary_by_train_exp_df
    

#%% Misc helpers
def get_train_experiment_names(experiment_names, delimiter="___"):
    """ 
    Create train experiment names by only considering the part of the experiment name before the "___" (or a different delimiter)
    """
    train_experiment_names = [x.split(delimiter)[0] for x in experiment_names]
    train_experiment_names = list(dict.fromkeys(train_experiment_names)) # creates list with unique elements but same order
    return train_experiment_names

# =============================================================================
# # maybe this is required later, but not now, for sure...
# def load_summary_file(experiment_name, file_name, n, results_base_dir):
#     # read summary file
#     if n == 1: # only one seed
#         summary_path = os.path.join(results_base_dir, experiment_name, "tables", file_name)
#         df = pd.read_csv(summary_path)
#     return df
# =============================================================================

#%%


# =============================================================================
# # for dev
# summary_stats_by_train_exp(experiment_names)
# 
# =============================================================================
