import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
import os
import numpy as np
import time

from misc_utils import get_aucroc
from storage_utils import load_best_model_state_dict, save_statistics


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
            
            ### validation set
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
        
        
        
class AnomalyDetectionExperiment(nn.Module):
    
    def __init__(self, experiment_name, anomaly_detection_experiment_name,
                 model, device,
                 test_data_loader, test_dataset,
                 measure_of_anomaly, window_aggregation_method, save_anomaly_maps, use_gpu):
        
        super(AnomalyDetectionExperiment, self).__init__()          
        
        self.measure_of_anomaly=measure_of_anomaly
        self.window_aggregation_method=window_aggregation_method
        self.save_anomaly_maps=save_anomaly_maps
        self.test_data_loader=test_data_loader
        self.test_dataset = test_dataset # This is needed to get the full size ground truth images
        self.test_image_list = test_dataset.image_list 
        self.test_image_sizes = test_dataset.image_sizes
        self.model = model
        self.device = device
        
        if torch.cuda.device_count() > 1:
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu
        
        # Load state dict from  best epoch of that experiment
        model_dir = os.path.abspath(os.path.join("results", experiment_name, "saved_models"))
        trained_as_parallel_AD_single_process = True if torch.cuda.device_count() < 1 else False
        state_dict = load_best_model_state_dict(model_dir=model_dir, use_gpu=use_gpu, trained_as_parallel_AD_single_process=trained_as_parallel_AD_single_process)
        self.load_state_dict(state_dict=state_dict["network"]) # Note: You need to load the state dict for the whole AnomalyDetection object, not just the model, since that is the format the state dict was saved in
        
        self.anomaly_map_dir = os.path.abspath(os.path.join("results", "anomaly_detection", experiment_name + "___" + anomaly_detection_experiment_name, "anomaly_maps"))
        if not os.path.exists(self.anomaly_map_dir):
            os.makedirs(self.anomaly_map_dir)

        self.result_tables_dir = os.path.abspath(os.path.join("results", "anomaly_detection", experiment_name + "___" + anomaly_detection_experiment_name, "tables"))
        if not os.path.exists(self.result_tables_dir):
            os.makedirs(self.result_tables_dir)


    def run_experiment(self):        
        self.model.eval()
       
        num_finished_images = -1 # the data loader works through the test set images in order. num_finished_images is a counter that ticks up everytime one image is finished
        self.stats_dict = {"aucroc":[]} # a dict that keeps the measures of agreement between pixel-wise anomaly score and ground-truth labels, for each image. Current,y AUC is the only measure.
        
        with tqdm.tqdm(total=len(self.test_data_loader)) as pbar:
            for inputs, targets, image_idxs, slices in self.test_data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model.forward(inputs)
                if self.measure_of_anomaly == "absolute distance":
                    anomaly_score = torch.abs(outputs - targets) # pixelwise anomaly score, for each of the images in the batch
                    anomaly_score = torch.mean(anomaly_score, dim=1, keepdim=True) # take the mean over the channels
                    anomaly_score = anomaly_score.cpu().detach()
                
                # the following for-loop deals with translating the pixelwise anomaly for one sliding window position (and thus a score relative to an image patch) into an anomaly score for the full image
                for batch_idx in range(len(image_idxs)): # for each image in the batch. "in range(batch_size)" leads to error, because the last batch is smaller that the batch size
                    # Get the index of the full image that the current patch was taken from. The index is relative to image_list and image_sizes
                    current_image_idx = int(image_idxs[batch_idx].detach().numpy())
                    assert num_finished_images <= current_image_idx, "This assertion fails if the dataloader does not strucutre the batches so that the order of images/patches WITHIN the batch does still correspond to image_list" # Basically, I am sure that __getitem__() gets items in the right order, but I am unsure if the order gets imxed up within the minibatch by the DataLoader. Probably best to leave that assertion in, since this will throw a bug if the behaviour of DataLoader is changed in future PyTorch versions.
            
                    
                    if current_image_idx > num_finished_images: # Upon starting the with the first patch, or whenever we have moved on to the next image
                        num_finished_images += 1
                        if num_finished_images > 0: # Whenever we have moved to the next image, calculate agreement between our anomaly score and the ground truth segmentation. (Obviously don't do this when we are jstus tarting with the first patch)
                            anomaly_map = self.normalise_anomaly_map(anomaly_map,normalisation_map)
                            self.calculate_agreement_between_anomaly_score_and_labels(
                                    image_idx=current_image_idx-1, anomaly_map=anomaly_map)
                            
#                            save_statistics(experiment_log_dir=self.result_tables_dir, filename='summary.csv',
#                            stats_dict=self.stats_dict, current_epoch=current_image_idx-1, continue_from_mode=True, save_full_dict=False) # save statistics to stats file.

                            
                            if self.save_anomaly_maps:
                                anomaly_map = transforms.functional.to_pil_image(anomaly_map)
                                anomaly_map.save(os.path.join(self.anomaly_map_dir,self.test_image_list[current_image_idx -1]))
                            
                        # Upon starting the with the first patch, or whenever we have moved on to the next image, create new anomaly maps and normalisation maps
                        current_image_height = self.test_image_sizes[current_image_idx][1]
                        current_image_width = self.test_image_sizes[current_image_idx][2]
                        
                        anomaly_map = torch.zeros((1,current_image_height, current_image_width)) # anomaly score heat maps for every image. Initialise as constant zero tensor of the same size as the full image
                        normalisation_map = torch.zeros((1,current_image_height, current_image_width)) # for every image, keep score of how often a given pixel has appeared in a sliding window, for calculation of average scores. Initialise as constant zero tensor of the same size as the full image
                    
                    # Now the part that happens for every image-patch(!): update the relevant part of the current anomaly_score map:
                    current_slice = np.s_[:,
                                          np.s_[slices["1_start"][batch_idx]:slices["1_stop"][batch_idx]],
                                          np.s_[slices["2_start"][batch_idx]:slices["2_stop"][batch_idx]]]
            
                    if self.window_aggregation_method == "mean":
                        anomaly_map[current_slice] += anomaly_score[batch_idx,:,:,:]
                        normalisation_map[current_slice] += 1
                
                # update progress bar
                pbar.update(1)
            
            
            # also calculate results and save anomaly map for the last image
            anomaly_map = self.normalise_anomaly_map(anomaly_map,normalisation_map)
            self.calculate_agreement_between_anomaly_score_and_labels(
                    image_idx=current_image_idx, anomaly_map=anomaly_map)
            
            save_statistics(experiment_log_dir=self.result_tables_dir, filename='summary.csv',
                            stats_dict=self.stats_dict, current_epoch=current_image_idx, continue_from_mode=False, save_full_dict=True) # save statistics to stats file.

            if self.save_anomaly_maps:
                anomaly_map = transforms.functional.to_pil_image(anomaly_map)
                anomaly_map.save(os.path.join(self.anomaly_map_dir,self.test_image_list[current_image_idx]))
                            
            # print mean results:
            print("Results:")
            for key, list_of_values in self.stats_dict.items():
                mean_value = sum(list_of_values)/len(list_of_values)
                print("Mean ", key, ": ", "{:.4f}".format(mean_value))
        
            

    
        
    def normalise_anomaly_map(self, anomaly_map, normalisation_map):
        if self.window_aggregation_method == "mean": # how we normalise the anomaly_map might depend on the window aggregation method
        # normalise anomaly score maps
            normalisation_map[normalisation_map == 0] = 1 # change zeros in the normalisation factor to 1
            anomaly_map = anomaly_map / normalisation_map
        return anomaly_map
    
    def calculate_agreement_between_anomaly_score_and_labels(self, image_idx, anomaly_map):
    
        # load ground truth segmentation label image
        label_image = self.test_dataset.get_label_image(image_idx)
        
        ### calculate measures of agreement 
        # AUC: currently the only measure of agreement
        if self.measure_of_anomaly == "absolute distance": #then all anomly scores will be in [0,1], so no further preprocessing is needed to calculate AUC:
            aucroc = get_aucroc(label_image, anomaly_map)
        
        self.stats_dict["aucroc"].append(aucroc)

    

# =============================================================================
# 
# ### normalise anomaly score  maps
# for image_idx in range(len(image_list)):
#     normalisation_maps[image_idx][normalisation_maps[image_idx] == 0] = 1 # change zeros in the normalisation factor to 1
#     anomaly_maps[image_idx] = anomaly_maps[image_idx] / normalisation_maps[image_idx]
#     
# 
# 
# # =============================================================================
# # 
# # ### combine anomaly scores - REPLACED BECAUSE OF MEMORY ISSUES
# # all_anomaly_scores = {}
# # for image_name, image_size in zip(image_list, image_sizes):
# #     if window_aggregation_method == "mean":
# #         combined_score_tensor = torch.zeros(image_size)
# #         windows_per_pixel = torch.zeros(image_size) # counts how many times a pixel appears in a window, for averaging
# #         for window_info in all_windows[image_name]:
# #             score_tensor = torch.zeros(image_size)
# #             score_tensor[window_info["slice relative to full image"]] = window_info["anomaly score"]
# #             combined_score_tensor = combined_score_tensor  + score_tensor
# #             windows_per_pixel[window_info["slice relative to full image"]] += 1
# #         windows_per_pixel[windows_per_pixel == 0] = 1
# #         combined_score_tensor = combined_score_tensor / windows_per_pixel
# #         all_anomaly_scores[image_name] = combined_score_tensor
# #         
# # =============================================================================
# ### testing
# show_idx = 1
# anomaly_score = anomaly_maps[show_idx].detach().numpy()
# anomaly_score = np.squeeze(anomaly_score)
# normalisation_map = normalisation_maps[show_idx].detach().numpy()
# normalisation_map = np.squeeze(normalisation_map)
# 
# plt.figure()
# plt.imshow(anomaly_score)
# plt.figure()
# plt.imshow(normalisation_map)
# 
# #%%
# ### Compare anomaly heat map with ground-truth labels:
# from sklearn.metrics import roc_auc_score
# def get_aucroc(y_true, output):
#     if torch.min(y_true.data) == 1 or torch.max(y_true.data) == 0:
#         aucroc = np.nan # return nan if there are only examples of one type in the batch, because AUCROC is not defined then. 
#     else:
#         y_true = y_true.cpu().detach().numpy().flatten()
#         output = output.cpu().detach().numpy().flatten()
#         aucroc = roc_auc_score(y_true,output)
#     return aucroc
# 
# 
# aucroc_per_image = np.empty(len(image_list))
# for idx, anomaly_map in enumerate(anomaly_maps):
#     label_image = data.get_label_image(idx)
#     
#     if measure_of_anomaly == "absolute distance": #then all anoamly scores will be in [0,1]:
#         aucroc_per_image[idx] = get_aucroc(label_image, anomaly_map)
#         
# aucroc_mn = np.mean(aucroc_per_image)
#         
#     
# =============================================================================
