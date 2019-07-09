"""A python script that automates job submission and provides functionality for constraining the total number of jobs submitted to the cluster at any given time. 
It also ensures jobs keep running, by periodically checking the queue, and resubmitting a job if it can not be found in the queue.

Also, I don't really understand it well.


"""

import os
import subprocess
import argparse
import tqdm
import getpass
import time

# Parse commandline input
parser = argparse.ArgumentParser(description='Welcome to the run N at a time script')
parser.add_argument('--num_parallel_jobs', type=int, help="This is the number of maximum jobs to have in the queue at any time")
parser.add_argument('--total_epochs', type=int, help="Number of epochs to run all jobs for")
args = parser.parse_args()


def check_if_experiment_with_name_is_running(experiment_name): # I haven't tested or understood this, but apparently it checks if an experiment with that name is already running
    result = subprocess.run(['squeue --name {}'.format(experiment_name), '-l'], stdout=subprocess.PIPE, shell=True)
    lines = result.stdout.split(b'\n')
    if len(lines) > 2:
        return True
    else:
        return False

student_id = getpass.getuser().encode()[:5] # get username in binary format
list_of_scripts = [item for item in
                   subprocess.run(['ls'], stdout=subprocess.PIPE).stdout.split(b'\n') if
                   item.decode("utf-8").endswith(".sh")] # list of all bash files. I don't know why you need to decode here

for script in list_of_scripts:
    print('sbatch', script.decode("utf-8"))

# dictionary with all script names as keys and value 0
epoch_dict = {key.decode("utf-8"): 0 for key in list_of_scripts}
total_jobs_finished = 0

while total_jobs_finished < args.total_epochs * len(list_of_scripts): # why does this multiply with total epochs here?
    curr_idx = 0 # index of current job
    with tqdm.tqdm(total=len(list_of_scripts)) as pbar_experiment: #just for visualisation
        while curr_idx < len(list_of_scripts):
            
            # detect how many jobs are currently in the queue
            number_of_jobs = 0 # number of jobs that are currently running
            result = subprocess.run(['squeue', '-l'], stdout=subprocess.PIPE) # run squeue -l and capture stdout
            for line in result.stdout.split(b'\n'):
                if student_id in line:
                    number_of_jobs += 1 # everytime my student-id appears, count up the number of jobs running
            
            
            if number_of_jobs < args.num_parallel_jobs: # if there is room to submit more jobs
                while check_if_experiment_with_name_is_running(
                        experiment_name=list_of_scripts[curr_idx].decode("utf-8")) or epoch_dict[
                    list_of_scripts[curr_idx].decode("utf-8")] >= args.total_epochs: # while experiment is already running or has run for over args.total_epochs
                    curr_idx += 1 # move to next script
                    if curr_idx >= len(list_of_scripts): # start at the beginning
                        curr_idx = 0

                str_to_run = 'sbatch {}'.format(list_of_scripts[curr_idx].decode("utf-8"))
                total_jobs_finished += 1
                os.system(str_to_run)
                print(str_to_run)
                curr_idx += 1
            else:
                time.sleep(1)