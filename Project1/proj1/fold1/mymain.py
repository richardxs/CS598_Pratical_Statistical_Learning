# Step 0: Load necessary Python packages
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# The following features are either highly imbalanced or not informative
remove_features_set = ['Condition_2', 'Heating', 'Latitude', 'Longitude', 'Low_Qual_Fin_SF',
                       'Misc_Feature', 'Pool_Area', 'Pool_QC', 'Roof_Matl', 'Street', 'Utilities']

winsor_features_set = ['BsmtFin_SF_2', 'Bsmt_Unf_SF', 'Enclosed_Porch', 'First_Flr_SF',
                       'Garage_Area', 'Gr_Liv_Area', 'Lot_Area', 'Lot_Frontage', 'Mas_Vnr_Area',
                       'Misc_Val', 'Open_Porch_SF', 'Screen_Porch', 'Second_Flr_SF', 'Three_season_porch',
                       'Total_Bsmt_SF', 'Wood_Deck_SF']


# categorical_feature_set = None

# Common Utility Method

def get_categorical_feature_set(df):
    categorical_feature_set = [feature for feature in df.columns if df[feature].dtypes == 'object']
    return categorical_feature_set


def categorical_variable_transform(df, categorical_feature_set=None, feature_encoder_mapping=None):
    # IMPORTANT:
    # The test_dataframe needs to use the encoder from the trainng_dataframe, because some categories might be
    # missing in the test data

    if not categorical_feature_set:
        categorical_feature_set = [feature for feature in df.columns if df[feature].dtypes == 'object']


    if not feature_encoder_mapping:
        feature_encoder_mapping = dict()


    for feature in categorical_feature_set:

        category_matrix = [[element] for element in df[feature]]
        if feature in feature_encoder_mapping:
            """
                if feature is present in the feature_encoder_mapping, use the encoding (This will be the case during the 
                test prediction time.
            """
            encoder = feature_encoder_mapping[feature]
        else:
            """
            During the training time the feature will not be available in the feature_encoder_mapping, therefore create an 
            entry with the fitted encoder
            """
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoder.fit(category_matrix)
            feature_encoder_mapping[feature] = encoder

        df_hot_code = pd.DataFrame(encoder.transform(category_matrix).toarray())

        df_hot_code.columns = [feature + '_' + str(c) for c in df_hot_code.columns]

        # Replace the original feature with one-hot encoded feature
        df.drop(columns=feature, inplace=True)
        df = pd.concat([df, df_hot_code], axis=1)

    return df, categorical_feature_set, feature_encoder_mapping


# Step 1: Preprocess the training data and fit the models
def preprocess_and_fit_models(train_csv):
    print(
        f"preprocess_and_fit_models(): Preprocessing the Training data and generating two fitted models Lasso and XGBoost.")
    # Load the training data
    df_train = pd.read_csv(train_csv)
    print(f"preprocess_and_fit_models(): df_train{df_train.head()}")


    # Split the data into features (X) and target labels (y)
    df_train_y = pd.DataFrame()
    df_train_y['Sale_Price'] = df_train['Sale_Price'].copy()
    df_train_x = df_train.drop(columns=['PID','Sale_Price'])  # 'PID',

    # Preprocess data (e.g., handle missing values, encode categorical features, scale, etc.)
    # Impute missing values in "Garage_Yr_Blt" variable with 0
    df_train_x['Garage_Yr_Blt'].fillna(0, inplace=True)

    df_train_x.drop(columns=remove_features_set, inplace=True)

    for val in winsor_features_set:
        upper_limit = df_train[val].quantile(0.974)
        df_train_x[val] = df_train_x[val].apply(lambda x: upper_limit if x > upper_limit else x)

    # Treating the two features "Mo_Sold" (1~12), and "Year_Sold" (2006~2010) as categorical
    # variables can improve model performance
    # df_train_x['Mo_Sold'] = df_train_x['Mo_Sold'].values.astype('object')
    # df_train_x['Year_Sold'] = df_train_x['Year_Sold'].values.astype('object')

    df_train_x_trans, categorical_feature_set, feature_encoder_mapping = categorical_variable_transform(df_train_x)

    # Log-scale sale_price
    df_train_y['Sale_Price'] = df_train_y['Sale_Price'].apply(lambda y: np.log(y))

    # Initialize and fit the first model : LASSO
    best_alpha = 0.0026048905108264305
    best_lasso = Lasso(fit_intercept=True, random_state=0, alpha=best_alpha, max_iter=10000)
    model_1 = Pipeline(steps=[("scalar", StandardScaler()), ("lasso", best_lasso)])


    model_1.fit(df_train_x_trans, df_train_y)

    # Initialize and fit the second model : XGBoost
    best_xgb = XGBRegressor(max_depth=2, n_estimators=420)
    model_2 = Pipeline(steps=[("scalar", StandardScaler()), ("xgb", best_xgb)])

    model_2.fit(df_train_x_trans, df_train_y)
    print(
        f"preprocess_and_fit_models(): categorical_feature_set: \n----------\n{categorical_feature_set}\n")

    return categorical_feature_set, feature_encoder_mapping, model_1, model_2


# Step 2: Preprocess test data and save predictions
def preprocess_and_save_predictions(test_csv, categorical_feature_set, feature_encoder_mapping, model_1, model_2):
    print(
        f"preprocess_and_save_predictions(): Preprocess test data and save predictions\n")
    # Load the test data
    df_test_x = pd.read_csv(test_csv)
    pid_df = pd.DataFrame()
    pid_df['PID'] =  df_test_x['PID'].copy()

    # Preprocess the test data to match the preprocessing applied to the training data
    df_test_x.drop(columns=['PID'], inplace=True)  # 2023_10_13
    df_test_x['Garage_Yr_Blt'].fillna(0, inplace=True)
    df_test_x.drop(columns=remove_features_set, inplace=True)

    for val in winsor_features_set:
        upper_limit = df_test_x[val].quantile(0.974)
        df_test_x[val] = df_test_x[val].apply(lambda x: upper_limit if x > upper_limit else x)

    # Treating the two features "Mo_Sold" (1~12), and "Year_Sold" (2006~2010) as categorical
    # variables can improve model performance
    # df_test_x['Mo_Sold'] = df_test_x['Mo_Sold'].values.astype('object')
    # df_test_x['Year_Sold'] = df_test_x['Year_Sold'].values.astype('object')

    df_test_x_trans, categorical_feature_set, feature_encoder_mapping = categorical_variable_transform(df_test_x,
                                                                                                       categorical_feature_set,
                                                                                                       feature_encoder_mapping)

    # Make predictions using the first model

    print(
        f"preprocess_and_save_predictions(): Generating Predictions based on the test_data sample: \n----\n {df_test_x_trans.head()}")
    predictions1 = model_1.predict(df_test_x_trans)

    # Make predictions using the second model
    predictions2 = model_2.predict(df_test_x_trans)

    # Create a DataFrame with PID and Sale_Price columns
    submission_df1 = pd.DataFrame({'PID': pid_df['PID'], 'Sale_Price': np.exp(predictions1)})
    submission_df2 = pd.DataFrame({'PID': pid_df['PID'], 'Sale_Price': np.exp(predictions2)})


    # Save predictions to two submission files in the specified format
    submission_df1.to_csv('mysubmission1.txt', index=False, sep=',', float_format='%.1f')
    submission_df2.to_csv('mysubmission2.txt', index=False, sep=',', float_format='%.1f')
    print(
        f"preprocess_and_save_predictions(): Generated files mysubmission1.txt, mysubmission2.txt with the predictions.")


if __name__ == "__main__":
    print("\n\n\n============[ PROCESS STARTS HERE !!]=================\n")
    """
    As mentioned under the "Code Evaluation" section, source script will be executed in a directory
    containing the two files train.csv & test.csv
    """
    train_csv = 'train.csv'  # path of the train.csv file
    test_csv = 'test.csv'  # path of the test.csv file

    print("STEP 1: PREPROCESS TRAINING AND FIT MODELS:\n=============================\n")

    categorical_feature_set, feature_encoder_mapping, model_1, model_2 = preprocess_and_fit_models(train_csv)
    print("\n\nSTEP 2: PREPROCESS TEST DATA AND SAVE PREDICTIONS:\n=============================\n")
    preprocess_and_save_predictions(test_csv, categorical_feature_set, feature_encoder_mapping, model_1, model_2)

    print("\n\n\n============[ PROCESS ENDS HERE !!]=================\n")
