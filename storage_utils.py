import pickle
import os
import csv
import torch
from collections import OrderedDict

def save_to_stats_pkl_file(experiment_log_filepath, filename, stats_dict):
    summary_filename = os.path.join(experiment_log_filepath, filename)
    with open("{}.pkl".format(summary_filename), "wb") as file_writer:
        pickle.dump(stats_dict, file_writer)


def load_from_stats_pkl_file(experiment_log_filepath, filename):
    summary_filename = os.path.join(experiment_log_filepath, filename)
    with open("{}.pkl".format(summary_filename), "rb") as file_reader:
        stats = pickle.load(file_reader)

    return stats


def save_statistics(experiment_log_dir, filename, stats_dict, current_epoch, continue_from_mode=False, save_full_dict=False):
    """
    Saves the statistics in stats dict into a csv file. Using the keys as the header entries and the values as the
    columns of a particular header entry
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file
    :param stats_dict: the stats dict containing the data to be saved
    :param current_epoch: the number of epochs since commencement of the current training session (i.e. if the experiment continued from 100 and this is epoch 105, then pass relative distance of 5.)
    :param save_full_dict: whether to save the full dict as is overriding any previous entries (might be useful if we want to overwrite a file)
    :return: The filepath to the summary file
    """
    summary_filename = os.path.join(experiment_log_dir, filename)
    mode = 'a' if continue_from_mode else 'w'
    with open(summary_filename, mode, newline='') as f:
        writer = csv.writer(f)
        if not continue_from_mode:
            writer.writerow(list(stats_dict.keys()))

        if save_full_dict:
            total_rows = len(list(stats_dict.values())[0])
            for idx in range(total_rows):
                row_to_add = [value[idx] for value in list(stats_dict.values())]
                writer.writerow(row_to_add)
        else:
            row_to_add = [value[current_epoch] for value in list(stats_dict.values())]
            writer.writerow(row_to_add)

    return summary_filename


def load_statistics(experiment_log_dir, filename):
    """
    Loads a statistics csv file into a dictionary
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file to load
    :return: A dictionary containing the stats in the csv file. Header entries are converted into keys and columns of a
     particular header are converted into values of a key in a list format.
    """
    summary_filename = os.path.join(experiment_log_dir, filename)

    with open(summary_filename, 'r+') as f:
        lines = f.readlines()

    keys = lines[0].split(",")
    stats = {key: [] for key in keys}
    for line in lines[1:]:
        values = line.split(",")
        for idx, value in enumerate(values):
            stats[keys[idx]].append(value)

    return stats


def update_state_dict_keys(state_dict):
    # Modify keys in a model state dict to use a model that was serialised as a nn.DataParallel module:
    # delete the .model prefix from the keys in the state dict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k != "network":
            new_state_dict[k] = v
    new_state_dict["network"] = {}
    for k, v in state_dict["network"].items():
        name = k.replace("model.", "") # remove `model.`
        name = name.replace("module.", "") # remove `module.`
        new_state_dict["network"][name] = v
    return new_state_dict

def load_best_model_state_dict(model_dir, use_gpu):
    # load the state dict of the model with the best validation performance (file name ends in _best)
    
    # find best model
    model_list = os.listdir(model_dir)
    for model_name in model_list:
        if model_name.endswith("_best"):
            best_model_name = model_name
            
    # load best model's state dict
    if use_gpu: # The models were all trained on GPU with DataParallel. When loading the state_dict on a CPU, we need to specify the map_location, and also rename the keys to handle the fact that this is not a DataParallel model any more.
        state_dict = torch.load(f = os.path.join(model_dir, best_model_name))
    else: # if loading on cpu, specify map location and modify keys to account for the fact that we won't use nn.DataParallel
        state_dict = torch.load(f = os.path.join(model_dir, best_model_name), map_location="cpu")
        state_dict = update_state_dict_keys(state_dict)
    
    return state_dict



