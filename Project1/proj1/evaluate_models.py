import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
import time
import os

def lasso_rmse(df_train_x_trans, df_test_x_trans, df_train_y, df_test_y, best_alpha):
  best_lasso = Lasso(fit_intercept=True, random_state=0, alpha=best_alpha, max_iter=10000)
  best_lasso_pipeline = Pipeline(steps=[("scalar",StandardScaler()), ("lasso", best_lasso)])

  best_lasso_pipeline.fit(df_train_x_trans , df_train_y)

  lasso_train_predict = best_lasso_pipeline.predict(df_train_x_trans)
  lasso_test_predict = best_lasso_pipeline.predict(df_test_x_trans)

  lasso_rmse_train = np.sqrt(np.mean((lasso_train_predict - np.squeeze(df_train_y.values))**2))
  lasso_rmse_test = np.sqrt(np.mean((lasso_test_predict - np.squeeze(df_test_y.values))**2))
  return lasso_rmse_test


def XGBoost_rmse(df_train_x_trans, df_test_x_trans, df_train_y, df_test_y, max_depth, n_estimators):
  best_xgb = XGBRegressor(max_depth=2, n_estimators=420)
  best_xgb_pipeline = Pipeline(steps=[("scalar",StandardScaler()), ("xgb", best_xgb)])

  best_xgb_pipeline.fit(df_train_x_trans , df_train_y)
  xgb_train_predict = best_xgb_pipeline.predict(df_train_x_trans)
  xgb_test_predict = best_xgb_pipeline.predict(df_test_x_trans)
  xgb_rmse_train = np.sqrt(np.mean((xgb_train_predict - np.squeeze(df_train_y.values))**2))
  xgb_rmse_test = np.sqrt(np.mean((xgb_test_predict - np.squeeze(df_test_y.values))**2))
  return xgb_rmse_test


def eval_lasso(y_lasso, test_y_csv):
    df_y_lasso = pd.read_csv(y_lasso)
    #df_y_lasso = pd.read_csv(y_lasso)
    df_test_y = pd.read_csv(test_y_csv)
    #print(f"df_y_lasso: \n{df_y_lasso.head(10)}")
    # Convert the DataFrame to a dictionary
    df_pred = df_y_lasso.head(10)
    print(f"df_pred: \n{df_pred}")
    df_y = df_test_y.head(10)
    print(f"==================")
    print(f"df_y: \n{df_y}")
    print(f"==================")

    pred_dict = df_pred.set_index('PID')['Sale_Price'].to_dict()
    y_dict = df_y.set_index('PID')['Sale_Price'].to_dict()
    print(f"pred_dict: \n --------\n{pred_dict}")
    print(f"==================")
    print(f"pred_dict: \n --------\n{y_dict}")
    print(f"==================")
    # Create NumPy arrays from the dictionary values
    y_true = np.array([y_dict[pid] for pid in pred_dict.keys()])
    y_pred = np.array([pred_dict[pid] for pid in pred_dict.keys()])
    # Calculate the mean squared error (MSE)
    #mse = mean_squared_error(y_true, y_pred)
    print(f"y_true: \n --------\n{y_true}")
    print(f"==================")
    print(f"y_pred: \n --------\n{y_pred}")
    print(f"==================")

    # Compute the RMSE
    #rmse = np.sqrt(mse)
    rmse = np.sqrt(np.mean((y_pred - np.squeeze(y_true)) ** 2))

    print(f"Root Mean Squared Error (RMSE): {rmse}")

def generate_rmse(data_folder, pred_file, true_val_file):
    df_pred = pd.read_csv(os.path.join(data_folder, pred_file))
    df_true_y = pd.read_csv(os.path.join(data_folder, true_val_file))


    pred_dict = df_pred.set_index('PID')['Sale_Price'].to_dict()
    true_y_dict = df_true_y.set_index('PID')['Sale_Price'].to_dict()

    # Create NumPy arrays from the dictionary values
    y_true = np.array([true_y_dict[pid] for pid in true_y_dict.keys()])
    y_pred = np.array([pred_dict[pid] for pid in pred_dict.keys()])

    rmse = np.sqrt(np.mean((np.log(y_pred) - np.squeeze(np.log(y_true))) ** 2))
    return rmse

    #pass

if __name__ == "__main__":
    print("\n\n\n============[ Evaluation PROCESS STARTS HERE !!]=================\n")

    train_csv = 'train.csv'  # path of the train.csv file
    test_csv = 'test.csv'  # path of the test.csv file
    test_y_csv = 'test_y.csv'  # path of the test.csv file

    y_lasso_csv = "mysubmission1.txt"
    y_xgb_csv = "mysubmission2.txt"

    root = 'fold'
    lasso_rmse_list = []
    tree_rmse_list = []
    #start_time = time.time()
    #for i in range(1, 9):
    i = 9
    lasso_rmse_list.append(generate_rmse(root+str(i),y_lasso_csv,test_y_csv))
    tree_rmse_list.append(
    generate_rmse(root+str(i),y_xgb_csv,test_y_csv))
    #end_time = time.time()
    #print(end_time - start_time)

    #lasso_rmse = eval_lasso(y_lasso_csv,test_y_csv)

    print("RMSE FOR LASSO:\n=============================\n")
    for rmse in lasso_rmse_list:
        print("%.4f" % rmse)
    print("RMSE FOR XGB:\n=============================\n")
    for rmse in tree_rmse_list:
        print("%.4f" % rmse)

    print("\n\n\n============[ PROCESS ENDS HERE !!]=================\n")
