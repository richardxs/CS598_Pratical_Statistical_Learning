# mymain.py

import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LassoCV, Lasso, RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

def load_data(train_path, test_path, vocab_path):

    train_data = pd.read_csv(train_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')

    with open(vocab_path, 'r') as f:
        vocabulary = f.read().splitlines()

    return train_data, test_data, vocabulary

def preprocess_training_data(df_train):
    print("preprocess_training_data(): Preprocessing training data")
    # Clean Review texts
    df_train['review'] = df_train['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

    positive_indices = df_train[df_train['sentiment'] == '1'].index.values
    negative_indices = df_train[df_train['sentiment'] == '0'].index.values
    num_pos = len(positive_indices)
    num_neg = len(negative_indices)

    return df_train,positive_indices,negative_indices,num_pos,num_neg


def construct_document_term_matrix(df_train_cleaned):
    stop_words = ["i", "me", "my", "myself",
                 "we", "our", "ours", "ourselves",
                 "you", "your", "yours",
                 "their", "they", "his", "her",
                 "she", "he", "a", "an", "and",
                 "is", "was", "are", "were",
                 "him", "himself", "has", "have",
                 "it", "its", "the", "us", "br"]

    vectorizer = CountVectorizer(
        preprocessor=lambda x: x.lower(),  # Convert to lowercase
        stop_words=stop_words,  # Remove stop words
        ngram_range=(1, 4),  # Use 1- to 4-grams
        min_df=0.001,  # Minimum term frequency
        max_df=0.5,  # Maximum document frequency
        token_pattern=r"\b[\w+\|']+\b"  # Use word tokenizer: See Ethan's comment below
    )

    dtm_train = vectorizer.fit_transform(df_train_cleaned['review'])

    # Retrieve features
    df_features_names = pd.DataFrame(vectorizer.get_feature_names_out(), columns=['feature_names'])
    return dtm_train,df_features_names

def preprocess_data(train_data, test_data, vocabulary):
    print("preprocess_data(): Preprocessing data")
    df_train_cleaned, positive_indices, negative_indices, num_pos, num_neg = preprocess_training_data(train_data)
    print(f"df_train_cleaned:{df_train_cleaned.shape}, num_pos:{num_pos}, num_neg: {num_neg}")
    print(f"")
    dtm_train, df_features_names = construct_document_term_matrix(df_train_cleaned)
    print(f"dtm_train: {dtm_train.shape}")
    print(f"df_features_names: {len(df_features_names)}")
    #------------------------------
    vectorizer = CountVectorizer(
        ngram_range=(1, 4),               # Use 1- to 4-grams
        vocabulary=vocabulary
    )

    X_train = vectorizer.transform(df_train_cleaned['review'])
    y_train = df_train_cleaned['sentiment']

    X_test = vectorizer.transform(test_data['review'])

    return X_train, y_train, X_test


def train_model(X_train, y_train):
    # Find Best Alpha
    alphas = np.linspace(1, 10, 20)
    ridge_model = LogisticRegression(penalty='l2', solver='liblinear')
    ridge_clf = GridSearchCV(ridge_model, [{'C': alphas}], cv=10, refit=False, scoring='roc_auc')
    ridge_clf.fit(X=X_train, y=y_train)
    best_alpha = ridge_clf.best_params_['C']

    print(f"ridge_clf.best_score_: {ridge_clf.best_score_}, best_alpha:{best_alpha}")

    # Create a model based on best alpha
    best_ridge_model = LogisticRegression(penalty='l2', solver='liblinear', C=best_alpha)
    best_ridge_model.fit(X=X_train, y=y_train)

    return best_ridge_model


def generate_submission(model, X_test, output_path):
    predictions = model.predict_proba(X_test)[:, 1]
    submission_df = pd.DataFrame({'id': test_data['id'], 'prob': predictions})
    submission_df.to_csv(output_path, index=False)


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

    X_train, y_train, X_test = preprocess_data(train_data, test_data, vocabulary)

    # Train model
    print("STEP 3: GENERATING MODEL:\n=============================\n")

    model = train_model(X_train, y_train)

    # Generate submission
    print("STEP 4: GENERATING SUBMISSION FILE:\n=============================\n")

    generate_submission(model, X_test, output_path)

    print("\n=============================\n \tPROJECT 3 EXECUTION FINISHED:\n=============================\n")

