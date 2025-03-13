import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
torch.manual_seed(42)
# Use ray tune specifically to tune the functions
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hebo import HEBOSearch


search_space = {
            "epochs": tune.choice([*range(40, 100, 5)]), 
            "batch": tune.choice([2, 4, 8, 16, 32]), 
            "lr0": tune.uniform(0.00001, 0.01),
            "momentum": tune.uniform(0.85, 0.99),
            "weight_decay": tune.uniform(0.00001, 0.1),
            "optimizer": tune.choice(["SGD", "AdamW", "Adam"])
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


def run_model_optimize(data_set, config):
    optimizer = torch.optim.Adam(MLR_model.parameters(), lr=config['lr'])
    # defining the loss criterion
    criterion = torch.nn.MSELoss()
    # Creating the dataloader
    train_loader = DataLoader(dataset=data_set, batch_size=config['batch_size'])
    # Train the model
    losses = []
    for epoch in range(config['epochs']):
        for x,y in train_loader:
            y_pred = MLR_model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
        print(f"epoch = {epoch}, loss = {loss}")
        losses.append(loss.item())
    print("Done training!")
 
# Create the dataset class
class Data():
    # Constructor
    def __init__(self,x, y):
        self.x = torch.tensor(np.array(x)).to(torch.float32)
        self.y = torch.tensor(np.array(y)).to(torch.float32)
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
        y_pred2 = F.relu(self.linear2(y_pred1))
        y_pred = self.dropout(self.linear3(y_pred2))
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
    # Create the data set object
    data_set = Data(train_x, train_y)
    # Creating the model object
    MLR_model = MultipleLinearRegression(len(train_x.columns), len(train_y.columns))
    print("The parameters: ", list(MLR_model.parameters()))
     
    optimizer = torch.optim.Adam(MLR_model.parameters(), lr=0.0001)
    # defining the loss criterion
    criterion = torch.nn.MSELoss()
     
    # Creating the dataloader
    train_loader = DataLoader(dataset=data_set, batch_size=4)
     
    # Train the model
    losses = []
    epochs = 100
    for epoch in range(epochs):
        for x,y in train_loader:
            y_pred = MLR_model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
        print(f"epoch = {epoch}, loss = {loss}")
        losses.append(loss.item())
    print("Done training!")
     
    # Plot the losses
    plt.plot(losses)
    plt.xlabel("no. of iterations")
    plt.ylabel("total loss")
    plt.show()