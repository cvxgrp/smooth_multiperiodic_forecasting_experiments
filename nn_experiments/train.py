import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
torch.manual_seed(42)
 
# Creating the dataset class
class Data():
    # Constructor
    def __init__(self,x, y):
        self.x = torch.tensor(np.array(x)).to(torch.float32)
        self.y = torch.tensor(np.array(y)).to(torch.float32)
        self.len = self.x.shape[0]
    # Getter
    def __getitem__(self, idx):          
        return self.x[idx], self.y[idx] 
    # getting data length
    def __len__(self):
        return self.len
 
# Creating a custom Multiple Linear Regression Model
class MultipleLinearRegression(torch.nn.Module):
    # Constructor
    def __init__(self, input_dim, output_dim):
        super(MultipleLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    # Prediction
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
 


if __name__ == '__main__':
    # Read in the training dataframes
    train_x = pd.read_csv("C:/Users/kperry/Documents/source/repos/smooth_multiperiodic_forecasting_experiments/X_in_sample.csv",
                           parse_dates=True,
                           index_col=0)
    train_y = pd.read_csv("C:/Users/kperry/Documents/source/repos/smooth_multiperiodic_forecasting_experiments/Y_in_sample.csv",
                          parse_dates=True,
                          index_col=0)
    # Create the data set object
    data_set = Data(train_x, train_y)
    # Creating the model object
    MLR_model = MultipleLinearRegression(len(train_x.columns), len(train_y.columns))
    print("The parameters: ", list(MLR_model.parameters()))
     
    # defining the model optimizer
    optimizer = torch.optim.SGD(MLR_model.parameters(), lr=0.0001)
    # defining the loss criterion
    criterion = torch.nn.MSELoss()
     
    # Creating the dataloader
    train_loader = DataLoader(dataset=data_set, batch_size=2)
     
    # Train the model
    losses = []
    epochs = 20
    for epoch in range(epochs):
        for x,y in train_loader:
            y_pred = MLR_model(x)
            loss = criterion(y_pred, y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
        print(f"epoch = {epoch}, loss = {loss}")
    print("Done training!")
     
    # Plot the losses
    plt.plot(losses)
    plt.xlabel("no. of iterations")
    plt.ylabel("total loss")
    plt.show()