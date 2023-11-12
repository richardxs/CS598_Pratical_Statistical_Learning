# Step 0: Load necessary Python packages
import os

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder


"""
   Project 2: Weekly Sales prediction for Walmart Stores
   =========
   Team Members: 
        1> Shu Xu (shuxu3@illinois.edu): Draft implementation
        2> Yan Han (yanhan4@illinois.edu): Code standardization , Report Generation
        3> Amrit Kumar(amritk2@illinois.edu): Executable Script Generation
   
   This script takes train.csv and test.csv as an input and produces "mypred.csv" as an output

"""
def transform_train_data_with_year_week(dataframe: pd.DataFrame):
    new_df = dataframe.copy()
    new_df['Year'] = new_df['Date'].dt.isocalendar().year
    new_df['Week'] = new_df['Date'].dt.isocalendar().week
    new_df.drop(columns=['Date', 'IsHoliday'], inplace=True)

    # combine year and week for svd smoothing
    new_df['Year_Week'] = new_df['Year'] * 100 + new_df['Week']
    new_df.drop(columns=['Year', 'Week'], inplace=True)

    # Re-order column names
    return new_df[['Store', 'Dept', 'Year_Week', 'Weekly_Sales']]


def group_data(dateframe: pd.DataFrame,
               keys: list = ['Store', 'Dept']):
    df_grouped = dateframe.groupby(keys)
    group_ids = []
    groups = []
    for id in df_grouped.groups:
        group_ids.append(id)
        groups.append(df_grouped.get_group(id).drop(columns=keys))

    return group_ids, groups

def smoothing_train_data_with_svd(dataframe: pd.DataFrame,
                                  num_pc: int = 8):
    ## fill out missing values/clean outliers
    group_ids, groups = group_data(dataframe, keys=['Dept'])

    svd_smoothed_df_list = []
    for i in range(len(group_ids)):
        df_i = groups[i].pivot(index='Store', columns='Year_Week', values='Weekly_Sales').fillna(0)
        mean_i = df_i.mean(axis=1).values
        df_i_values_centered = (df_i.values.T - mean_i).T
        U, S, Vh = np.linalg.svd(df_i_values_centered, full_matrices=False)
        new_S = np.diag(S[:num_pc])

        df_svd_smoothed_values = ((U[:, :num_pc] @ new_S @ Vh[:num_pc, :]).T + mean_i).T

        df_svd_smoothed = pd.DataFrame(df_svd_smoothed_values,
                                       index=df_i.index,
                                       columns=df_i.columns).reset_index()

        df_i_svd_smoothed_unpivot = pd.melt(df_svd_smoothed,
                                            id_vars='Store',
                                            value_vars=df_svd_smoothed.columns). \
            sort_values(by=['Store', 'Year_Week'])

        df_i_svd_smoothed_unpivot['Year'] = (df_i_svd_smoothed_unpivot['Year_Week'].values // 100).astype(int)
        df_i_svd_smoothed_unpivot['Week'] = (df_i_svd_smoothed_unpivot['Year_Week'].values % 100).astype(int)
        df_i_svd_smoothed_unpivot.rename(columns={'value': 'Weekly_Sales'}, inplace=True)
        df_i_svd_smoothed_unpivot.drop(columns=['Year_Week'], inplace=True)
        df_i_svd_smoothed_unpivot.reset_index(drop=True)
        df_i_svd_smoothed_unpivot['Dept'] = group_ids[i]
        svd_smoothed_df_list.append(df_i_svd_smoothed_unpivot)

    svd_smoothed_df = pd.concat(svd_smoothed_df_list).reset_index(drop=True)
    holiday_weeks = [6, 36, 47, 52]
    svd_smoothed_df['IsHoliday'] = np.where(svd_smoothed_df['Week'].isin(holiday_weeks), True, False)

    # Keep column order consistent with original one
    return svd_smoothed_df[['Store', 'Dept', 'Year', 'Week', "IsHoliday", 'Weekly_Sales']]


def categorical_variable_transform(train_df, test_df):
    # IMPORTANT:
    # The test_dataframe needs to use the encoder from the trainng_dataframe, because some categories might be
    # missing in the test data
    categorical_feature_set = [feature for feature in train_df.columns if train_df[feature].dtypes == 'object']
    new_train_df = train_df.copy()
    new_test_df = test_df.copy()
    for feature in categorical_feature_set:
        encoder = OneHotEncoder(handle_unknown='ignore')
        train_category_matrix = [[element] for element in train_df[feature]]
        test_category_matrix = [[element] for element in test_df[feature]]

        encoder.fit(train_category_matrix)
        train_df_hot_code = pd.DataFrame(encoder.transform(train_category_matrix).toarray())
        test_df_hot_code = pd.DataFrame(encoder.transform(test_category_matrix).toarray())

        # Different from Project#1, add 1 here
        train_df_hot_code.columns = [feature + '_' + str(c + 1) for c in train_df_hot_code.columns]
        test_df_hot_code.columns = [feature + '_' + str(c + 1) for c in test_df_hot_code.columns]

        # Replace the original feature with one-hot encoded feature
        new_train_df.drop(columns=feature, inplace=True)
        new_train_df = pd.concat([new_train_df, train_df_hot_code], axis=1)

        new_test_df.drop(columns=feature, inplace=True)
        new_test_df = pd.concat([new_test_df, test_df_hot_code], axis=1)

    # Further feature engineering
    new_train_df.drop(columns='Weekly_Sales', inplace=True)
    new_train_df['Year'] = new_train_df['Year'] - 2010

    new_test_df['Year'] = new_test_df['Year'] - 2010

    return new_train_df, new_test_df


def transform_test_data_with_year_week(dataframe: pd.DataFrame):
    new_df = dataframe.copy()
    new_df['Year'] = new_df['Date'].dt.isocalendar().year
    new_df['Week'] = new_df['Date'].dt.isocalendar().week
    new_df.drop(columns=['Date'], inplace=True)

    # Re-order column names
    return new_df[['Store','Dept', 'Year', 'Week', 'IsHoliday']]


def predict(df_train_svd_smoothed, df_train_one_hot_encode, df_test_one_hot_encode, alpha=5):
    train_Y_group_ids, train_Y_groups = group_data(df_train_svd_smoothed[['Store', 'Dept', 'Weekly_Sales']])
    train_X_group_ids, train_X_groups = group_data(df_train_one_hot_encode, keys=['Store', 'Dept'])
    test_X_group_ids, test_X_groups = group_data(df_test_one_hot_encode, keys=['Store', 'Dept'])

    prediction_results = []
    for i, index_pair in enumerate(test_X_group_ids):
        if index_pair not in train_X_group_ids:
            pred_test = np.zeros(len(test_X_groups[i]))
        else:
            i_train = train_X_group_ids.index(index_pair)

            temp_train_X = train_X_groups[i_train]
            temp_train_Y = train_Y_groups[i_train]
            temp_test_X = test_X_groups[i]

            # The training X are basically 1 and 0 s, no need to stanardize
            lasso_model = Lasso(alpha=alpha)
            lasso_model.fit(temp_train_X, temp_train_Y)
            pred_test = lasso_model.predict(temp_test_X)
            if i <= 5:
                print(f"type: {type(pred_test)} , pred_test_{i} : {pred_test}")

        prediction_results.extend(list(pred_test))

    return np.array(prediction_results)


def preprocess_training_data(df_train):
    print(f" preprocess_training_data(): Preprocessing Training Data")
    df_train_transformed = transform_train_data_with_year_week(df_train)
    print(f"preprocess_training_data(): df_train_transformed:\n {df_train_transformed.head()}")

    df_train_svd_smoothed = smoothing_train_data_with_svd(dataframe=df_train_transformed)
    print(f"preprocess_training_data(): df_train_svd_smoothed:\n {df_train_svd_smoothed.head()}")

    df_train_svd_smoothed['Week'] = df_train_svd_smoothed['Week'].astype('object')
    df_train_svd_smoothed['IsHoliday'] = df_train_svd_smoothed['IsHoliday'].astype('object')

    return df_train_svd_smoothed


def predict_sales(train_csv, test_csv):
    """

    :param train_csv: Training data from historical Walmart sales
    :return: predicted sales
    """
    print("STEP 1: CALLING PREPROCESSING LOGIC:\n=============================\n")
    print(f"predict_sales(): Preprocessing the Training data.")

    df_train = pd.read_csv(train_csv, parse_dates=['Date'])
    print(f"predict_sales(): df_train:\n {df_train.head()}")


    df_train_svd_smoothed = preprocess_training_data(df_train)

    # Preprocess the test data
    print(f"predict_sales(): Preprocessing the Test data.")
    df_test = pd.read_csv(test_csv, parse_dates=['Date'])
    df_test_transform = transform_test_data_with_year_week(df_test)
    
    # Get the

    print(f"predict_sales(): Generating encoded Training and  Test data.")

    df_train_one_hot_encode, df_test_one_hot_encode = categorical_variable_transform(df_train_svd_smoothed,
                                                                                     df_test_transform)
    print("STEP 2: CALLING PREDICTION LOGIC:\n=============================\n")
    print(f"predict_sales(): Generating Predicted Data.")
    alpha = 5
    weekly_pred = predict(df_train_svd_smoothed, df_train_one_hot_encode, df_test_one_hot_encode, alpha)

    df_test['Weekly_Pred'] = weekly_pred.tolist()

    print("STEP 3: GENERATING PREDICTION FILE:\n=============================\n")

    print(f"predict_sales(): Generating \"mypred.csv\" with the predicted data.")

    df_test.to_csv('mypred.csv', index=False, sep=',',float_format='%.2f')

    print(f"predicted labels : {df_test.head()}")
    return df_test

if __name__ == "__main__":
    print("\n\n\n============[ PROJECT 2 PROCESSING STARTS HERE !!]=================\n")
    """
    As mentioned under the "Code Evaluation" section, source script will be executed in a directory
    containing the two files train.csv & test.csv
    """
    train_csv = 'train.csv'  # path of the train.csv file
    test_csv = 'test.csv'  # path of the test.csv file

    print("STEP 0: CALLING PREPROCESSING AND PREDICTION LOGIC:\n=============================\n")
    predict_sales(train_csv, test_csv)


    print("\n\n\n============[ PROCESS ENDS HERE !!]=================\n")
