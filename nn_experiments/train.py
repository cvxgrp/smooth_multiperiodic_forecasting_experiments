import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
torch.manual_seed(42)
# Use ray tune specifically to tune the functions
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hebo import HEBOSearch
import torch.optim as optim

# Use CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

search_space = {
            "epochs": tune.choice([*range(40, 100, 5)]), 
            "batch": tune.choice([2, 4, 8, 16, 32]), 
            "lr0": tune.uniform(0.00001, 0.01),
            "optimizer": tune.choice(["SGD", "AdamW", "Adam"]),
            "dropout_rate": tune.uniform(0.1, 0.5) 
        }

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


def run_model(train_data_set, val_data_set, config, x_dim, y_dim):
    # Declare the model
    model = MultipleLinearRegression(x_dim, y_dim, config['dropout_rate']).to(device)
    optimizer = getattr(optim, config["optimizer"].capitalize())(model.parameters(),
                                                                 lr=config["lr0"])
    # defining the loss criterion
    criterion = torch.nn.MSELoss()
    # Creating the dataloader
    train_loader = DataLoader(dataset=train_data_set, batch_size=config['batch_size'])
    test_loader = DataLoader(dataset=val_data_set, batch_size=config['batch_size'])
    epoch_performance_list = list()
    # Training and Evaluation loop
    for epoch in range(config['epochs']):
        print('EPOCH {}:'.format(epoch + 1))
        model.train() 
        avg_loss = 0.0
        for i, dataset in enumerate(train_loader):
            # Every data instance is an input + label pair
            data, target = dataset
            # Move to GPU
            data, target= data.to(device), target.to(device)
            optimizer.zero_grad()  # Clear gradients from the previous iteration
            output = model(data)  # Forward pass through the model
            loss = criterion(output, target)  # Calculate the loss
            loss.backward()  # Compute gradients (backpropagation)
            optimizer.step()  # Update model parameters
            avg_loss += loss.item()
            if i % 1000 == 999:
                last_loss = avg_loss / 1000
                running_loss = 0.
                    
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
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
        self.linear1 = torch.nn.Linear(input_dim, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(self, x):
        y_pred1 = F.relu(self.linear1(x))
        y_pred2 = self.dropout(F.relu(self.linear2(y_pred1)))
        y_pred = self.linear3(y_pred2)
        return y_pred
 


if __name__ == '__main__':
    # Read in the training dataframes
    train_x = pd.read_csv("C:/Users/kperry/Documents/source/repos/smooth_multiperiodic_forecasting_experiments/X_in_sample.csv",
                           parse_dates=True,
                           index_col=0)
    train_y = pd.read_csv("C:/Users/kperry/Documents/source/repos/smooth_multiperiodic_forecasting_experiments/Y_in_sample.csv",
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
    config = {"batch_size": 8,
              "lr0": .001, 
              "epochs": 20,
              "optimizer": "Adam",
              "dropout_rate": 0.01}
    # Run the model (example case)
    run_model(train_data_set, val_data_set, config, len(train_x.columns), len(train_y.columns))
    # # Run hyperparameter optimization with ray-tune
    # tuner = tune.Tuner(
    #             train_model,
    #             param_space=search_space,
    #             tune_config=tune.TuneConfig(search_alg=optuna_search, 
    #                                         num_samples=20)
    #             )
    # tuner.fit()