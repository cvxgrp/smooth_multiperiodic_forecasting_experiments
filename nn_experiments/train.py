import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
torch.manual_seed(42)
 
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
    
    def __init__(self, input_dim, output_dim):
        super(MultipleLinearRegression, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 1024)
        self.linear2 = torch.nn.Linear(1024, 512)
        self.linear3 = torch.nn.Linear(512, 256)
        self.linear4 = torch.nn.Linear(256, output_dim)
        
    def forward(self, x):
        y_pred1 = self.linear1(x)
        y_pred2 = self.linear2(y_pred1)
        y_pred3 = self.linear3(y_pred2)
        y_pred = self.linear4(y_pred3)
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
     
    # Use the Adam optimizer (SGD has exploding gradients issue)
    optimizer = torch.optim.Adam(MLR_model.parameters(), lr=0.0001)
    # defining the loss criterion
    criterion = torch.nn.MSELoss()
     
    # Creating the dataloader
    train_loader = DataLoader(dataset=data_set, batch_size=2)
     
    # Train the model
    losses = []
    epochs = 40
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