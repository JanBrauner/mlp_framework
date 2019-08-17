import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='ticks', palette='Set2')

# parameters

# =============================================================================
# # for dev
# experiment_names = ['CE_DTD_r2_stand_scale_1___AD_window_mean',
# 'CE_DTD_r2_stand_scale_1___AD_window_min']
# 
# experiment_name = 'CE_DTD_r2_stand_scale_1___AD_window_mean'
# 
# =============================================================================

# paths
results_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "results", "anomaly_detection"))

# feature_dict = parse_experiment_name(experiment_name, features)


def summary_stats(experiment_names, features, sort_column=None, display_table=True, results_base_dir=results_base_dir):
    """
    Create and print DataFrame that contains the stats (mean + SD) of several measures from several experiments.
    Inputs csv is/are expected to contain columns with column names + values.
    The experiments should have the same measures (although it will probably do something useful othewise as well)
    feature set is used for parse_experiment_name, see there.
    """
    summary_df = pd.DataFrame(columns=["experiment_name", "val_mean_aucroc", 
                                       "val_sd", "val_sem", "test_mean_aucroc", 
                                       "test_sd", "test_sem", "val_images_analysed", "test_images_analysed"]) # predefine columns to have the df in that order
    
    for experiment_name in experiment_names:
        curr_exp_stats = {"experiment_name": experiment_name} # stats dicts
        curr_exp_stats.update(parse_experiment_name(experiment_name, features))
        
        for which_set in ["val", "test"]:
            
            # load df
            file_name = which_set + "_summary.csv"
            summary_path = os.path.join(results_base_dir, experiment_name, "tables", file_name)
            df = pd.read_csv(summary_path)
            
            # calculate relevant stats
            mn = df.loc[:,"aucroc"].mean()
            sd = df.loc[:,"aucroc"].std()
            sem = df.loc[:,"aucroc"].sem()
            num = len(df)
            
            # update stats_dict
            curr_exp_stats["{}_mean_aucroc".format(which_set)] = mn
            curr_exp_stats["{}_sd".format(which_set)] = sd
            curr_exp_stats["{}_sem".format(which_set)] = sem
            curr_exp_stats["{}_images_analysed".format(which_set)] = num
            

        
        summary_df = summary_df.append(curr_exp_stats, ignore_index=True)    
    
    summary_df = summary_df.set_index("experiment_name")
    
    if sort_column is not None: # sort for displaying
        summary_df = summary_df.sort_values(by=sort_column, ascending=False)
    if display_table:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(summary_df)
    return summary_df



### I probably don't need this function anymore, since parse_experiment_name is much better and in the spirit of pandas. 
### If you just create additional columns, you can use pd functionality like filtering, pivoting, ...
### But let's keep this function around for now
def summary_stats_by_train_exp(experiment_names, delimiter="___", display_table=True):
    """
    Summary table over multiple experiments, but all AD experiments that go back to the same train experiment are in one row.
    Note: 
        this could be made much more flexible, so that results could be grouped by other parts of the experiment_name as well.
        !! But a much better solution would be to read out the experiment_name strings at the beginning and transfer into columns
    """
    summary_df = summary_stats(experiment_names, display_table=False)
    
    train_experiment_names = get_train_experiment_names(summary_df.index, delimiter=delimiter)
    
    # not sure if this is important...
    summary_by_train_exp_df = pd.DataFrame(columns=["train_experiment_name"])
    summary_by_train_exp_df.set_index("train_experiment_name")
    
    
    for train_experiment_name in train_experiment_names:
        dict_to_append = {"train_experiment_name":train_experiment_name}
    
        for row in summary_df.itertuples(): # note that row is returned as a namedtuple
            if train_experiment_name in row.Index: # update the stats_dict with the stats of taht experiment, prepended by the AD experiment name. E.g. val_loss becomes <AD_experiment_val_loss>
                AD_experiment_name = row.Index.split(delimiter)[1]
                dict_to_append.update({(AD_experiment_name +"_" + k): v for k,v in row._asdict().items() if not "Index" in k})

        summary_by_train_exp_df = summary_by_train_exp_df.append(dict_to_append, ignore_index=True)
    
    summary_by_train_exp_df = summary_by_train_exp_df.set_index("train_experiment_name")
    
    if display_table:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            display(summary_by_train_exp_df)
    return summary_by_train_exp_df
    

#%% Misc helpers

def get_features(feature_set_name):
    if feature_set_name == "r2":
        features = [{"column_name": "Output type",
                     "patterns": ["stand", "prob"],
                     "values": ["deterministic", "probabilistic"]},
    
                    {"column_name": "Inpainting setting",
                     "patterns": ["scale_1", "scale_0p5", "small_mask", "large_context"],
                     "values": ["standard", "large mask", "small mask", "large patch"]},
                     
                    {"column_name": "Anomaly value aggregation function",
                     "patterns": ["window_min", "window_mean", "window_max"],
                     "values": ["minimum", "mean", "maximum"]},
                    
                    {"column_name": "Model",
                     "patterns": ['CE_DTD_r2_stand_scale',
                                  'CE_DTD_r2_stand_scale_0p5',
                                  'CE_DTD_r2_stand_small_mask',
                                  'CE_DTD_r2_stand_large_context',
                                  'CE_DTD_r2_prob_scale_1',
                                  'CE_DTD_r2_prob_scale_0p5',
                                  'CE_DTD_r2_prob_small_mask',
                                  'CE_DTD_r2_prob_large_context',],
                     "values": ["determ. standard", 
                                "determ. large mask", 
                                "determ. small mask", 
                                "determ. large patch", 
                                "probab. standard", 
                                "probab. large mask", 
                                "probab. small mask", 
                                "probab. large patch",]}
                    ]
    elif feature_set_name == "r2_comb":
        features = [{"column_name": "output_type",
                     "patterns": ["stand", "prob"],
                     "values": ["stand", "prob"]},
    
                    {"column_name": "inpainting_setting",
                     "patterns": ["Sc1Sc05", "Sc1Sm", "Sc1Lc", "Sc05Sm", "Sc05Lc", "SmLc", "Sc1Sc05Sm", "Sc1Sc05Lc", "Sc1SmLc", "Sc05SmLc", "Sc1Sc05SmLc"],
                     "values": ["Sc1Sc05", "Sc1Sm", "Sc1Lc", "Sc05Sm", "Sc05Lc", "SmLc", "Sc1Sc05Sm", "Sc1Sc05Lc", "Sc1SmLc", "Sc05SmLc", "Sc1Sc05SmLc"]},
                     
                    {"column_name": "window_aggregation_method",
                     "patterns": ["win_min", "win_mean", "win_max"],
                     "values": ["min", "mean", "max"]},
                    
                    {"column_name": "model_combination_method",
                     "patterns": ["comb_min", "comb_mean", "comb_max"],
                     "values": ["min", "mean", "max"]}
                    
                    ]
    elif feature_set_name == "r3":
        features = [{"column_name": "autoencoding_setting",
                      "patterns": ["patch_64", "patch_128", "full_image_128"],
                      "values": ["patch_64", "patch_128", "full_image_128"]},
        
                     {"column_name": "num_bottleneck",
                      "patterns": ["bn_8192", "bn_4096", "bn_2048", "bn_1024", "bn_512", "bn_256", "bn_128"],
                      "values": [8192, 4096, 2048, 1024, 512, 256, 128]},
                      
                     {"column_name": "window_aggregation_method",
                      "patterns": ["window_min", "window_mean", "window_max"],
                      "values": ["min", "mean", "max"]},
                      
                     {"column_name": "output_type",
                     "patterns": ["prob", "r3_patch", "r3_full"],
                     "values": ["prob", "stand", "stand"]},
                      ]
    elif feature_set_name == "r7":
        features = [{"column_name": "output_type",
                     "patterns": ["stand", "prob"],
                     "values": ["stand", "prob"]},
    
                    {"column_name": "inpainting_setting",
                     "patterns": ["scale_1", "scale_0p71", "scale_0p5", "scale_0p35", "scale_0p25", "scale_0p18", "scale_0p125", "small_mask", "large_context"],
                     "values": ["scale_1", "scale_0p71", "scale_0p5", "scale_0p35", "scale_0p25", "scale_0p18", "scale_0p125", "small_mask", "large_context"]},
                     
                    {"column_name": "window_aggregation_method",
                     "patterns": ["win_min", "win_mean", "win_max"],
                     "values": ["min", "mean", "max"]}]
    elif feature_set_name == "r8":
        features = [{"column_name": "autoencoding_setting",
                      "patterns": ["scale", "full_image"],
                      "values": ["patch", "full_image"]},
        
                     {"column_name": "num_bottleneck",
                      "patterns": ["bn_8192", "bn_4096", "bn_2048", "bn_1024", "bn_512", "bn_256", "bn_128", "bn_64"],
                      "values": [8192, 4096, 2048, 1024, 512, 256, 128, 64]},
                      
                     {"column_name": "window_aggregation_method",
                      "patterns": ["win_min", "win_mean", "win_max"],
                      "values": ["min", "mean", "max"]},
                      
                     {"column_name": "scale",
                      "patterns": ["scale_1", "scale_0p71", "scale_0p5", "scale_0p35", "scale_0p25", "scale_0p18", "scale_0p125"],
                      "values": [1, 0.71, 0.5, 0.35, 0.25, 0.18, 0.125]}
                      ]
    return features

# =============================================================================
#    features = [{"column_name": "output_type",
#              "patterns": ["stand", "prob"],
#              "values": ["stand", "prob"]},
#             {"column_name": "inpainting_setting",
#              "patterns": ["scale_1", "scale_0p5", "small_mask", "large_context"],
#              "values": ["scale_1", "scale_0p5", "small_mask", "large_context"]},
#             {"column_name": "window_aggregation_method",
#              "patterns": ["window_min", "window_mean", "window_max"],
#              "values": ["min", "mean", "max"]},
# #            {"column_name": ,
# #             "patterns": ,
# #             "values": },
# #            {"column_name": ,
# #             "patterns": ,
# #             "values": }
#     ]
# =============================================================================

def parse_experiment_name(experiment_name, features):
    """
    features can be a string describing a predefined feature set, or a list of dictionaries directly describing the features.
    features: a list of dictionaries. Each dict corresponds to one column that will be added to the df.
    Entries of each dict:
        "column_name": Name of the column to be added, string
        "patterns": possible patters that can be found in experiment_name, list of strings
        "values": list of values, where values[i] is the entry that should be added to the new column if patterns[i] is found in experiment_name
    
    """
    if type(features) == str:
        features = get_features(feature_set_name = features)
    feature_dict = {}
    for feature in features:
        for pattern, value in zip(feature["patterns"], feature["values"]):
            if pattern in experiment_name:
                feature_dict.update({feature["column_name"]: value})
    
    
    # additionally. get train and AD experiment_name
    feature_dict.update({"train_experiment_name": experiment_name.split("___")[0],
                         "AD_experiment_name": experiment_name.split("___")[1]
    })
    
    return feature_dict

### Probably don't need this anymore, since parse_experiment_name does the same and more.
### but let's keep it around for a while
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
