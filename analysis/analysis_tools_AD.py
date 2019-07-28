import os

import pandas as pd


# parameters





# paths
results_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "results", "anomaly_detection"))

def summary_stats_multi_exp(experiment_names, sort_column=None):
    summary_df = pd.DataFrame(columns=["experiment_name", "val_mean_aucroc", "val_sd", "test_mean_aucroc", "test_sd"])
    
    for experiment_name in experiment_names:
        curr_exp_stats = {"experiment_name": experiment_name}
        
        for which_set in ["val", "test"]:
            file_name = which_set + "_summary.csv"
            summary_path = os.path.join(results_base_dir, experiment_name, "tables", file_name)
            df = pd.read_csv(summary_path)
            mn = df.loc[:,"aucroc"].mean()
            sd = df.loc[:,"aucroc"].std()
            if which_set == "val":
                curr_exp_stats["val_mean_aucroc"] = mn
                curr_exp_stats["val_sd"] = sd
            elif which_set == "test":
                curr_exp_stats["test_mean_aucroc"] = mn
                curr_exp_stats["test_sd"] = sd
        
        summary_df = summary_df.append(curr_exp_stats, ignore_index=True)    
    
    summary_df.set_index("experiment_name")
    if sort_column is not None:
        summary_df.sort_values(by=sort_column)
    display(summary_df)

# =============================================================================
# # maybe this is required later, but not now, for sure...
# def load_summary_file(experiment_name, file_name, n, results_base_dir):
#     # read summary file
#     if n == 1: # only one seed
#         summary_path = os.path.join(results_base_dir, experiment_name, "tables", file_name)
#         df = pd.read_csv(summary_path)
#     return df
# =============================================================================
