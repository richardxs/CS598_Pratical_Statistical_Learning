# mymain.py

import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

"""
   Project 3: Movie Review Sentiment Analysis
   =========
   Team Members: 
        1> Shu Xu (shuxu3@illinois.edu): Draft implementation
        2> Yan Han (yanhan4@illinois.edu): Code standardization , Report Generation
        3> Amrit Kumar(amritk2@illinois.edu): Executable Script Generation

   Input:
   -------
        This script accepts myvocab.txt, train.tsv and test.tsv as inputs.

   Output:
   -------
        Generates file named mysubmission.csv with headers [id,     prob]

"""


def load_data(train_path, test_path, vocab_path):
    print(f"load_data(): Loading input files")
    train_data = pd.read_csv(train_path, sep='\t', header=0, dtype=str)
    test_data = pd.read_csv(test_path, sep='\t', header=0, dtype=str)

    with open(vocab_path, 'r') as f:
        vocabulary = f.read().splitlines()
    print(
        f"load_data(): Loading finished!! \n train_data: {train_data.shape} ,test_data: {test_data.shape} , vocabulary: {type(vocabulary)} -> {len(vocabulary)} ")
    return train_data, test_data, vocabulary


def train_model(dtm_train, y_train):
    print(f"train_model(): Initializing and training model.")
    ridge_model = LogisticRegression(penalty='l2', solver='liblinear', C=5.5)
    ridge_model.fit(X=dtm_train, y=y_train)
    return ridge_model


def generate_submission(model, X_test, output_path):
    print(f"generate_submission(): Generate predictions and submission file.")
    predictions = model.predict_proba(X_test)[:, 1]
    submission_df = pd.DataFrame({'id': test_data['id'], 'prob': predictions})
    submission_df.to_csv(output_path, index=False)


def get_vectorizer(myvocab):
    print(f"get_vectorizer(): Initializing and returns CountVectorizer based on input 'vocabulary'.")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        vocabulary=myvocab
    )

    return vectorizer


def preprocess_and_transform_data(train_data, test_data, vectorizer):
    print(f"preprocess_and_transform_data(): Preprocesses and transforms input data.")
    df_train = train_data
    df_train['review'] = df_train['review'].str.replace('<.*?>', '', regex=True)
    df_train['review'] = df_train['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)
    dtm_train = vectorizer.fit_transform(df_train['review'])
    train_y = df_train['sentiment']

    df_test_x = test_data
    df_test_x['review'] = df_test_x['review'].str.replace('<.*?>', '', regex=True)
    df_test_x['review'] = df_test_x['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)
    dtm_test = vectorizer.transform(df_test_x['review'])
    return dtm_train, train_y, dtm_test


if __name__ == "__main__":
    print("\n=============================\n \t:PROJECT 3 EXECUTION BEGINS:\n=============================\n")

    # Input paths
    train_path = "train.tsv"
    test_path = "test.tsv"
    vocab_path = "myvocab.txt"

    # Output path
    output_path = "mysubmission.csv"

    # Load data
    print("STEP 1: LOADING DATA:\n=============================\n")

    train_data, test_data, vocabulary = load_data(train_path, test_path, vocab_path)

    # Preprocess data
    print("STEP 2: CALLING PREPROCESSING LOGIC:\n=============================\n")
    vectorizer = get_vectorizer(vocabulary)
    dtm_train, train_y, dtm_test = preprocess_and_transform_data(train_data, test_data, vectorizer)
    # Train model
    print("STEP 3: GENERATING MODEL:\n=============================\n")

    model = train_model(dtm_train, train_y)

    # Generate submission
    print("STEP 4: GENERATING SUBMISSION FILE:\n=============================\n")

    generate_submission(model, dtm_test, output_path)

    print("\n=============================\n \tPROJECT 3 EXECUTION FINISHED:\n=============================\n")
