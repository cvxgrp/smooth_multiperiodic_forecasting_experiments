"""
Train an xgboost model to predict 24 hours out, using 7-day previous data.
"""

import pandas as pd
import xgboost as xgb
import glob
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import os

tune_model = False
data_folder_path = "C:/Users/kperry/Documents/source/repos/smooth_multiperiodic_forecasting_experiments/"

# Define the random search grid
parameters = {'learning_rate': [0.001, 0.01, 0.05, 0.1],
               'max_depth': [2, 5, 10, 15, 20, 25],
               'min_child_weight': [1, 5, 10, 15, 20, 25],
               'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
               'n_estimators': [200, 300, 400, 500, 600, 700, 800, 900, 1000],
               "reg_alpha": [0.5, 0.2, 1, 2, 5],
               "reg_lambda": [2, 3, 5, 10],
               "gamma": [1, 2, 3, 4, 5]}

tuned_hyperparameters = {'reg_lambda': 5, 
                          'reg_alpha': 0.2, 
                          'n_estimators': 700, 
                          'min_child_weight': 10, 
                          'max_depth': 5, 
                          'learning_rate': 0.01,
                          'gamma': 1, 
                          'colsample_bytree': 0.6}


def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

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

def reformat_data_to_regression(x, y):
    """
    Refactor the train/test data for a set to match what we'd expect as 
    an input to a regression problem.
    """
    # Pivot the test data accordingly
    y['date'] = pd.to_datetime(y.index)
    y_pivot = pd.melt(y, id_vars = ['date'])
    y_pivot['horizon_step_number'] = y_pivot['variable'].astype(int)
    # Order by date and then variable
    y_pivot = y_pivot.sort_values(by=['date', 'horizon_step_number'])
    y_pivot = y_pivot[['date', 'horizon_step_number', 'value']]
    # get the datetime associated with when the prediction is going to be made
    y_pivot['prediction_datetime'] = [(date + pd.DateOffset(hours = horizon_step_number))
                                            for date,horizon_step_number in zip(
                                                    y_pivot['date'], y_pivot['horizon_step_number'])] 
    # get the hour and day associated with the prediction time
    y_pivot['horizon_hour'] = y_pivot['prediction_datetime'].dt.hour
    y_pivot['horizon_day'] = y_pivot['prediction_datetime'].dt.day
    x['date'] = pd.to_datetime(x.index)
    # Join with the training data based on the start date of the forecast
    merged_master_df = pd.merge(x, y_pivot, on='date')
    merged_master_df.index =merged_master_df['date']
    # Convert horizon_hour and horizon_day values to cyclical encoded format
    merged_master_df = encode(merged_master_df,
                              col="horizon_hour",
                              max_val = 24)
    merged_master_df = encode(merged_master_df,
                              col="horizon_day",
                              max_val = 365)
    merged_master_df = merged_master_df[['horizon_step_number', 
                                         'horizon_hour_sin', 
                                         'horizon_hour_cos', 
                                         'horizon_day_sin',
                                         'horizon_day_cos',
                                         '0', '1', '2', '3',
                                         '4', '5', '6',
                                         '7', '8', '9', '10', '11', '12',
                                         '13', '14', '15', '16',
                                         '17', '18', '19', '20', 
                                         '21', '22', '23', '24',
                                         '25', '26', '27', '28', 
                                         '29', '30', '31', '32', 
                                         '33', '34', '35', '36',
                                         '37', '38', '39', '40',
                                         '41', '42', '43', '44', 
                                         '45', '46', '47', '48',
                                         '49', '50', '51', '52', 
                                         '53', '54', '55', '56', 
                                         '57', '58', '59', '60',
                                         '61', '62', '63', '64', 
                                         '65', '66', '67', '68', 
                                         '69', '70', '71', '72',
                                         '73', '74', '75', '76', 
                                         '77', 'value']]
    return merged_master_df
    
if __name__ == '__main__':
    # Read in the training and the test dataframes
    train_x = pd.read_csv(os.path.join(data_folder_path, "X_in_sample.csv"),
                           parse_dates=True,
                           index_col=0)
    train_y = pd.read_csv(os.path.join(data_folder_path, "Y_in_sample.csv"),
                          parse_dates=True,
                          index_col=0)
    # Ok let's perform our feature engineering. Keep all of Girays features,
    # and just add a couple columns for what time horizon we're predicting
    # for
    time_series = reformat_data_to_regression(train_x, train_y).dropna()
    # Loop through all of the X dataframe columns and min-max normalize.
    # save the normalization values
    normalization_params_list = list()
    for col in list(time_series):
        if col in ['horizon_step_number', 'horizon_hour_sin', 'horizon_hour_cos',
                   'horizon_day_sin', 'horizon_day_cos', '0', '1', '2', '3',
                   '4', '5']:
            pass
        else:
            min_val, max_val = min(time_series[col]), max(time_series[col])
            time_series[col] = min_max_normalize(min_val, max_val, time_series[col])
            normalization_params_list.append({"col": col,
                                              "min": min_val,
                                              "max": max_val})
    # Perform time series conversion into dependent and independent
    # variables
    X = time_series.drop(['value'], axis=1).values
    y = time_series['value'].values
    if tune_model:
        # Now let's tune using random search
        xgb_model = xgb.XGBRegressor()
        random_search_model = RandomizedSearchCV(xgb_model,
                                                 parameters,
                                                 n_iter=20,
                                                 cv=3,
                                                 scoring='neg_mean_absolute_error',
                                                 verbose=5,
                                                 n_jobs=2)
        random_search_model.fit(X, y, verbose=1)
        tuned_hyperparameters = random_search_model.best_params_
        print(tuned_hyperparameters)
        # Take the best hyperparameters and rerun the model with the best parameters
        xgb_model_optimized = xgb.XGBRegressor(**tuned_hyperparameters)
        xgb_model_optimized.fit(X, y, verbose=1)
        # Write the optimized model to memory
        xgb_model_optimized.save_model("optimized_xgboost_model.json")
    else:
        xgb_model_optimized = xgb.XGBRegressor(**tuned_hyperparameters)
        # Fit the model with the tuned hyperparameters
        xgb_model_optimized.fit(X, y, verbose=1)
        # Load in the test data
        test_x = pd.read_csv(os.path.join(data_folder_path, 
                                          "X_out_sample.csv"),
                               parse_dates=True,
                               index_col=0)
        test_y = pd.read_csv(os.path.join(data_folder_path,
                                          "Y_out_sample.csv"),
                              parse_dates=True,
                              index_col=0)
        # Clean up the data for the model
        test_time_series = reformat_data_to_regression(test_x, test_y)
        # Normalize with respect to the training data
        for col in list(test_time_series):
            if col in ['horizon_step_number', 'horizon_hour_sin', 'horizon_hour_cos',
                       'horizon_day_sin', 'horizon_day_cos', '0', '1', '2', '3',
                       '4', '5']:
                pass
            else:
                data_vals = [x for x in normalization_params_list if x["col"] == col][0]
                min_val, max_val = data_vals['min'], data_vals['max']
                test_time_series[col] = min_max_normalize(min_val, max_val,
                                                          test_time_series[col])
        # Predict for test set
        predict_X = test_time_series.drop(['value'], axis=1).values
        real_y = test_time_series['value'].values
        # Run the test data through the model to generate predictions
        predict_y = pd.Series(xgb_model_optimized.predict(predict_X))
        # Un-normalize the data w/ respect to the training data
        target_vals = [x for x in normalization_params_list if x["col"] == "value"][0]
        predict_y_unnormalized = ((predict_y * (target_vals['max'] -
                                               target_vals['min'])) +
                                  target_vals['min'])
        predict_y_unnormalized.index=test_time_series.index
        real_y_unnormalized = ((real_y * (target_vals['max'] -
                                               target_vals['min'])) +
                                  target_vals['min'])
        # Take the results and pivot them so that they're in the original 
        # expected format
        residuals = real_y_unnormalized - predict_y_unnormalized
        print("RESIDUALS RESULTS:")
        print("MAE:" + str(abs(residuals).mean()))
        print("Median ABS ERROR:" + str(abs(residuals).median()))    
        results_df = pd.DataFrame({"datetime": test_time_series.index,
                                    'horizon_step_number': test_time_series['horizon_step_number'],
                                   "predicted": predict_y_unnormalized})
        # Pivot the dataframe back 
        results_df_pivoted = results_df.pivot(index="datetime",
                                              columns="horizon_step_number")['predicted']
        results_df_pivoted.to_csv("xgboost_results.csv")