import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
# Use ray tune specifically to tune the functions
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hebo import HEBOSearch
import torch.optim as optim
import os


torch.manual_seed(42)

# Use CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameter search space
search_space = {
            "epochs": tune.choice([*range(20, 100, 5)]), 
            "batch_size": tune.choice([2, 4, 8, 16, 32]), 
            "lr0": tune.uniform(0.00001, 0.01),
            "optimizer": tune.choice(["Adam"]),
            "dropout_rate": tune.uniform(0.01, 0.5) 
        }

# Config of hyperparameters if we're just running the model
hyperparameter_config = {"batch_size": 16,
                         "lr0": 0.000288151, 
                         "epochs": 55,
                         "optimizer": "Adam",
                         "dropout_rate": 0.360359}

# Identify if we want to run hyperparameter tuning or not. If yes,
# identify the tuning strategy we want to use.
run_tuning = False
strategy = "random_grid_search"
number_runs = 10
data_folder_path = "C:/Users/kperry/Documents/source/repos/smooth_multiperiodic_forecasting_experiments/"

def min_max_normalize(min_val, max_val, data):
    """
    Min-max normalize with respect to the training data for a column

    Parameters
    ----------
    min_val : TYPE
        DESCRIPTION.
    max_val : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    data_normalized = (data - min_val) / (max_val - min_val)
    return data_normalized


def run_hyperparameter_tune_fn(config, train_data_set, val_data_set, x_dim, y_dim):
    """
    Optimization function for hyperparameter tuning via Ray Tune.
    """
    # Declare the model
    model = MultipleLinearRegression(x_dim, y_dim, config['dropout_rate']).to(device)
    optimizer = getattr(optim, config["optimizer"].capitalize())(model.parameters(),
                                                                 lr=config["lr0"])
    # defining the loss criterion
    criterion = torch.nn.MSELoss()
    # Creating the dataloader
    train_loader = DataLoader(dataset=train_data_set, batch_size=config['batch_size'])
    val_loader = DataLoader(dataset=val_data_set, batch_size=config['batch_size'])
    # Training and Evaluation loop
    for epoch in range(config['epochs']):
        model.train() 
        avg_loss = 0.0
        for i, dataset in enumerate(train_loader):
            # Every data instance is an input + label pair
            data, target = dataset
            data, target= data.to(device), target.to(device)
            optimizer.zero_grad()  # Clear gradients from the previous iteration
            output = model(data)  # Forward pass through the model
            loss = criterion(output, target)  # Calculate the loss
            loss.backward()  # Compute gradients (backpropagation)
            optimizer.step()  # Update model parameters
            avg_loss += loss.item()
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()
    
        avg_vloss = running_vloss / (i + 1)
    # Return the validation loss--this is what we're going to optimize against
    return {'val_loss': avg_vloss}    
    

def run_model(train_data_set, val_data_set, config, x_dim, y_dim):
    # Declare the model
    model = MultipleLinearRegression(x_dim, y_dim, config['dropout_rate']).to(device)
    optimizer = getattr(optim, config["optimizer"].capitalize())(model.parameters(),
                                                                 lr=config["lr0"])
    # defining the loss criterion
    criterion = torch.nn.MSELoss()
    # Creating the dataloader
    train_loader = DataLoader(dataset=train_data_set, batch_size=config['batch_size'])
    val_loader = DataLoader(dataset=val_data_set, batch_size=config['batch_size'])
    epoch_performance_list = list()
    # Training and Evaluation loop
    for epoch in range(config['epochs']):
        print('EPOCH {}:'.format(epoch + 1))
        model.train() 
        avg_loss = 0.0
        last_loss = 0.0
        for i, dataset in enumerate(train_loader):
            # Every data instance is an input + label pair
            data, target = dataset
            data, target= data.to(device), target.to(device)
            optimizer.zero_grad()  # Clear gradients from the previous iteration
            output = model(data)  # Forward pass through the model
            loss = criterion(output, target)  # Calculate the loss
            loss.backward()  # Compute gradients (backpropagation)
            optimizer.step()  # Update model parameters
            avg_loss += loss.item()
        # Compute last loss based on the number of batches
        last_loss = avg_loss / (i + 1)
                    
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()
    
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(last_loss, avg_vloss))
        epoch_performance_list.append({"epoch": epoch,
                                       "train_loss": last_loss,
                                       "val_loss": avg_vloss})
    # Create graphic of train and validation losses over epochs
    epoch_loss = pd.DataFrame(epoch_performance_list)
    epoch_loss[['train_loss', 'val_loss']].plot()
    plt.title("Training and Validation Loss over Epochs")
    plt.show()
    plt.close()
    return model


# Create the dataset class
class Data():
    # Constructor
    def __init__(self,x, y):
        self.x = torch.tensor(np.array(x)).to(torch.float32).to(device)
        self.y = torch.tensor(np.array(y)).to(torch.float32).to(device)
        self.len = self.x.shape[0]
    
    def __getitem__(self, idx):          
        return self.x[idx], self.y[idx] 

    def __len__(self):
        return self.len
 
# Create Multiple Linear Regression Model
class MultipleLinearRegression(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(MultipleLinearRegression, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(self, x):
        y_pred1 = F.relu(self.linear1(x))
        y_pred2 = self.dropout(F.relu(self.linear2(y_pred1)))
        y_pred3 = self.dropout(F.relu(self.linear3(y_pred2)))
        y_pred = self.linear4(y_pred3)
        return y_pred
 


if __name__ == '__main__':
    # Read in the training dataframes
    train_x = pd.read_csv(os.path.join(data_folder_path, "X_in_sample.csv"),
                           parse_dates=True,
                           index_col=0)
    train_y = pd.read_csv(os.path.join(data_folder_path, "Y_in_sample.csv"),
                          parse_dates=True,
                          index_col=0)
    # Perform min-max normalization on the data
    x_normalization_params_list = list()
    for col in list(train_x):
        min_val, max_val = min(train_x[col]), max(train_x[col])
        train_x[col] = min_max_normalize(min_val, max_val, train_x[col])
        x_normalization_params_list.append({"col": col,
                                            "min": min_val,
                                            "max": max_val})
    y_normalization_params_list = list()
    for col in list(train_y):
        min_val, max_val = min(train_y[col]), max(train_y[col])
        train_y[col] = min_max_normalize(min_val, max_val, train_y[col])
        y_normalization_params_list.append({"col": col,
                                            "min": min_val,
                                            "max": max_val})
    # Split the data set into training/validation
    cutoff_idx = round(len(train_x) *.8)
    val_x = train_x[cutoff_idx:]        
    val_y = train_y[cutoff_idx:]  
    train_x = train_x[:cutoff_idx]        
    train_y = train_y[:cutoff_idx]  
    # Create the data set object
    train_data_set = Data(train_x, train_y)
    val_data_set = Data(val_x, val_y)
    if run_tuning:
        if strategy == "random_grid_search":
            search_strategy = BasicVariantGenerator()
        if strategy == "optuna":
            search_strategy = OptunaSearch()
        if strategy == "hyperopt":
            search_strategy = HyperOptSearch()
        if strategy == "hebo":
            search_strategy = HEBOSearch()
        tuner = tune.Tuner(tune.with_resources(tune.with_parameters(
                            run_hyperparameter_tune_fn,
                            train_data_set=train_data_set,
                            val_data_set=val_data_set, 
                            x_dim=len(train_x.columns),
                            y_dim=len(train_y.columns)), 
                          {"gpu": 1, "cpu": 10}), 
                           tune_config=tune.TuneConfig(
                               metric='val_loss',
                               mode="min",
                               search_alg=search_strategy,
                               num_samples=number_runs,
                               max_concurrent_trials=1,
                               trial_dirname_creator=lambda trial: str(
                                   trial.trial_id)),
                           param_space=search_space)
        results = tuner.fit()
    else:
        # Run the model (example case)
        model = run_model(train_data_set, val_data_set, 
                          hyperparameter_config, 
                          len(train_x.columns), 
                          len(train_y.columns))
        # Read in the test data and pre-process it
        test_x = pd.read_csv(os.path.join(data_folder_path, "X_out_sample.csv"),
                               parse_dates=True,
                               index_col=0)
        test_y = pd.read_csv(os.path.join(data_folder_path, "Y_out_sample.csv"),
                              parse_dates=True,
                              index_col=0)    
        # Normalize all of the data w/r to the training data set
        for col in list(test_x):
            data_vals = [x for x in x_normalization_params_list if x["col"] == col][0]
            min_val, max_val = data_vals['min'], data_vals['max']
            test_x[col] = min_max_normalize(min_val, max_val, test_x[col])
        # Read data into data loader
        test_data_set = Data(test_x, test_y)
        test_loader = DataLoader(dataset=test_data_set, 
                                 batch_size=hyperparameter_config['batch_size'])
        # Generate associated predictions and unnormalize to original units
        model.eval()
        with torch.no_grad():
            predict_y = model(test_data_set.x)
        predict_y = pd.DataFrame(predict_y.cpu())
        # Un-transform and compare results to original test-y data
        for col in list(predict_y):
            data_vals = [x for x in y_normalization_params_list if x["col"] == str(col)][0]
            min_val, max_val = data_vals['min'], data_vals['max']
            predict_y[col] = (predict_y[col] * (max_val - min_val)) + min_val
        # calculate mean and median absolute error
        mean_absolute_error = abs(np.array(test_y) - np.array(predict_y)).mean()
        print("MAE: " + str(mean_absolute_error))
        median_absolute_error = np.median(abs(np.array(test_y) -
                                              np.array(predict_y)))
        print("Median Absolute Error: " + str(median_absolute_error))
        predict_y.index = test_y.index
        # write the results to a CSV
        predict_y.to_csv("nn_results.csv")