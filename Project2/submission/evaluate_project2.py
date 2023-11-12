import os

import numpy as np
import pandas as pd

import warnings

from mymain import predict_sales

warnings.filterwarnings("ignore", category=DeprecationWarning)

def evaluate(df_test, df_test_with_label):
    scoring_df = df_test.drop(columns=['IsHoliday']).merge(df_test_with_label, on=['Store', 'Dept','Date'], how='left')
    weights = scoring_df['IsHoliday'].apply(lambda x:5 if x else 1)
    actuals = scoring_df['Weekly_Sales']
    preds = scoring_df['Weekly_Pred']

    wae = np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)
    return wae

def main(train_file, test_file):
    # Your main logic here
    # For example, load the CSV files using pandas
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

def evaluate_10_fold(alpha=5):
    project_path = "../"
    test_with_label_file = os.path.join(project_path, "test_with_label.csv")
    print(f"test_with_label_file: {test_with_label_file}")
    df_test_with_label = pd.read_csv(os.path.join(project_path, "test_with_label.csv"), parse_dates=['Date'])
    wae_list= []
    print(f"df_test_with_label: {df_test_with_label.head()}")

    for fold_num in range(1, 11):
        fold = "fold_" + str(fold_num)
        # df_train = pd.read_csv(os.path.join(project_path, fold ,"train.csv"), parse_dates=['Date'])
        # df_test = pd.read_csv(os.path.join(project_path, fold, "test.csv"), parse_dates=['Date'])

        train_csv = os.path.join(project_path, fold ,"train.csv")
        test_csv = os.path.join(project_path, fold, "test.csv")

        print(f" training_csv_path: {train_csv} \n test_csv_path: {test_csv}")
        pred_df = predict_sales(train_csv, test_csv)

        wae = evaluate(pred_df, df_test_with_label)
        wae_list.append(wae)
    return wae_list


if __name__ == "__main__":
    print("\n\n\n============[ PROJECT 2 EVALUATION STARTS HERE !!]=================\n")
    wae_list = evaluate_10_fold(5)
    print("\n\n\n============[ PROJECT 2 EVALUATION RESULTS]=================\n")

    print(wae_list)
    print(sum(wae_list) / 10)

    print("\n\n\n============[ PROJECT 2 EVALUATION FINISHED !!]=================\n")
