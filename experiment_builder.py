import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time

from storage_utils import save_statistics

class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, train_data, val_data,
                 test_data, device, args):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = args.experiment_name
        self.model = network_model
        self.model.reset_parameters()
        self.device = device

        if torch.cuda.device_count() > 1:
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu
          # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optim.Adam(self.parameters(), amsgrad=False, lr=args.learning_rate, betas=args.betas,
                                    weight_decay=args.weight_decay_coefficient)
        self.task = args.task
        self.loss = args.loss
        # Generate the directory names
        self.experiment_folder = os.path.abspath(os.path.join("results", self.experiment_name))
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs)
        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        
        if self.task == "classification":
            self.best_val_model_measure = 0. # performance measure for choosing best epoch: accuracy
        elif self.task == "regression":
            self.best_val_model_measure = 1000000000 # performance measure for choosing best epoch: loss

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = args.num_epochs

#        # Antreas had this but I think it isn't needed if we use a functional loss anyway
#        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        
        if args.continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_measure, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best val model index
                # and the best val accuracy of that model
                self.starting_epoch = self.state['current_epoch_idx']
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()

        elif args.continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_measure, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=args.continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val accuracy of that model
            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0
            self.state = dict()

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def run_train_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        out, loss = self.forward_prop_and_loss(x,y)

        # update parameters
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss
        self.optimizer.step()  # update network parameters
        
        # return metrics
        if self.task == "classification":
            _, predicted = torch.max(out.data, 1)  # get argmax of predictions
            accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
            return loss.data.detach().cpu().numpy(), accuracy
        elif self.task == "regression":
            return loss.data.detach().cpu().numpy()

    def run_evaluation_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode

        out, loss = self.forward_prop_and_loss(x,y)

        # return metrics        
        if self.task == "classification":
            _, predicted = torch.max(out.data, 1)  # get argmax of predictions
            accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
            return loss.data.detach().cpu().numpy(), accuracy
        elif self.task == "regression":
            return loss.data.detach().cpu().numpy()

    
    def forward_prop_and_loss(self, x, y):
        # reshape inputs and targets
        if self.task == "classification":
            if len(y.shape) > 1:
                y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
            if type(x) is np.ndarray:
                x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
                device=self.device)  # convert data to pytorch tensors and send to the computation device
        elif self.task == "regression":
            if type(x) is np.ndarray:
                x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).float().to(
                device=self.device)  # convert data to pytorch tensors and send to the computation device

        x = x.to(self.device)
        y = y.to(self.device)
        out = self.model.forward(x)  # forward the data in the model
        
        if self.task == "classification":
            if self.loss == "cross-entropy":
                loss = F.cross_entropy(out, y)  # compute loss
        elif self.task == "regression":
            if self.loss == "L2":
                loss = F.mse_loss(out, y)
        
        
        return out, loss



    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_accuracy: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        state['network'] = self.state_dict()  # save network parameter and other variables.
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state OF THE LATEST EPOCH and the best val model idx and best val accuracy to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model accuracy, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_measure'], state

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        # initialize a dict to keep the per-epoch metrics
        if self.task == "classification":
            total_losses = {"train_accuracy": [], "train_loss": [], "val_accuracy": [],
                            "val_loss": [], "curr_epoch": []}
        elif self.task == "regression":
            total_losses = {"train_loss": [],
                        "val_loss": [], "curr_epoch": []}
            
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            
            if self.task == "classification":
                current_epoch_losses = {"train_accuracy": [], "train_loss": [], "val_accuracy": [], "val_loss": []}
            elif self.task == "regression":
                current_epoch_losses = {"train_loss": [], "val_loss": []}

            ### training set
            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for idx, (x, y) in enumerate(self.train_data):  # get data batches
                    if self.task == "classification":
                        loss, accuracy = self.run_train_iter(x=x, y=y)  # take a training iter step
                        self.update_current_epoch_stats(current_epoch_losses, current_dataset="train", loss=loss,accuracy=accuracy)
                        pbar_train.update(1)
                        pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
                    elif self.task == "regression":
                        loss = self.run_train_iter(x=x, y=y)  # take a training iter step
                        self.update_current_epoch_stats(current_epoch_losses, current_dataset="train", loss=loss)
                        pbar_train.update(1)
                        pbar_train.set_description("loss: {:.4f}".format(loss))

            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
                for x, y in self.val_data:  # get data batches
                    if self.task == "classification":
                        loss, accuracy = self.run_evaluation_iter(x=x, y=y)  # run a validation iter
                        self.update_current_epoch_stats(current_epoch_losses, current_dataset="val", loss=loss,accuracy=accuracy)
                        pbar_val.update(1)  # add 1 step to the progress bar
                        pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
                    elif self.task == "regression":
                        loss = self.run_evaluation_iter(x=x, y=y)  # run a validation iter
                        self.update_current_epoch_stats(current_epoch_losses, current_dataset="val", loss=loss)
                        pbar_val.update(1)  # add 1 step to the progress bar
                        pbar_val.set_description("loss: {:.4f}".format(loss))
            
            self.update_best_epoch_measure(current_epoch_losses, epoch_idx)
            
            ### validation set
            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(
                    value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            total_losses['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False) # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_measure'] = self.best_val_model_measure
            self.state['best_val_model_idx'] = self.best_val_model_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val accuracy, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val accuracy, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest', state=self.state)
        ### test set
        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        # load best validation model
                        model_save_name="train_model")
        
        if self.task == "classification":
            current_epoch_losses = {"test_accuracy": [], "test_loss": []}  # initialize a statistics dict
        elif self.task == "regression":
            current_epoch_losses = {"test_loss": []}  # initialize a statistics dict


        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
            for x, y in self.test_data:  # sample batch
                if self.task == "classification":
                    loss, accuracy = self.run_evaluation_iter(x=x, y=y)  # run a validation iter
                    self.update_current_epoch_stats(current_epoch_losses, current_dataset="test", loss=loss,accuracy=accuracy)
                    pbar_test.update(1)  # add 1 step to the progress bar
                    pbar_test.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
                elif self.task == "regression":
                    loss = self.run_evaluation_iter(x=x, y=y)  # run a validation iter
                    self.update_current_epoch_stats(current_epoch_losses, current_dataset="test", loss=loss)
                    pbar_test.update(1)  # add 1 step to the progress bar
                    pbar_test.set_description("loss: {:.4f}".format(loss))


        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)
        
        # rename best validation model file for easy access
        path_best = os.path.join(self.experiment_saved_models, "{}_{}".format("train_model", str(self.best_val_model_idx)))
        os.rename(path_best, path_best + "_best")
        
        return total_losses, test_losses
    
    
    def update_current_epoch_stats(self, current_epoch_losses, current_dataset, loss=[], accuracy = []):
        if self.task == "classification":
            current_epoch_losses["{}_loss".format(current_dataset)].append(loss)  # add current iter loss to the train loss list
            current_epoch_losses["{}_accuracy".format(current_dataset)].append(accuracy)  # add current iter accuracy to the train accuracy list
        elif self.task == "regression":
            current_epoch_losses["{}_loss".format(current_dataset)].append(loss)  # add current iter loss to the train loss list


    def update_best_epoch_measure(self, current_epoch_losses, epoch_idx):    
        """
        Updates statistics for best epoch, if the current epoch is the best epoch so far
        """
        if self.task == "classification":
            val_mean_performance_measure = np.mean(current_epoch_losses['val_accuracy']) # measure that determines which is the best epoch. For classification: accuracy                    
            if val_mean_performance_measure > self.best_val_model_measure:  # if current epoch's mean performance measure is better than the saved best one then
                self.best_val_model_measure = val_mean_performance_measure  # set the best val model accuracy to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx
                

        elif self.task == "regression":
            val_mean_performance_measure = np.mean(current_epoch_losses['val_loss']) # measure that determines which is the best epoch. For regression: loss                    
            if val_mean_performance_measure < self.best_val_model_measure:  # if current epoch's mean performance measure is better than the saved best one then
                self.best_val_model_measure = val_mean_performance_measure  # set the best val model accuracy to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx
        