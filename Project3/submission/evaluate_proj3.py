import os
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from Project3.submission.mymain import load_data,preprocess_and_transform_data, get_vectorizer,train_model

def evaluate_proj3():
    project_path = "../proj3_data/split_"

    # Input paths
    train_file = "train.tsv"
    test_file = "test.tsv"
    myvocab_file = "myvocab.txt"
    auc_score_list = []

    vocab_path = os.path.join("../", myvocab_file)
    with open(vocab_path, 'r') as f:
        vocabulary = f.read().splitlines()

    vectorizer = get_vectorizer(vocabulary)

    for fold_num in range(1, 6):
        fold = str(fold_num)
        data_path = project_path + str(fold)
        print("-----------------------------------------")
        print(f"fold_num:{fold_num} , data_path: {data_path}")
        train_path = os.path.join(data_path, train_file)
        test_path = os.path.join(data_path, test_file)
        print(f"train_path:{train_path} , test_path: {test_path}")
        train_data, test_data, vocabulary = load_data(train_path, test_path, vocab_path)


        dtm_train, train_y, dtm_test = preprocess_and_transform_data(train_data, test_data, vectorizer)
        time0 = time.time()
        model = train_model(dtm_train, train_y)

        time1 = time.time()
        y_proba = model.predict_proba(dtm_test)[:, 1]
        time2 = time.time()
        df_test_y = pd.read_csv(os.path.join(data_path, "test_y.tsv"), sep='\t', header=0, dtype=str)
        auc_score_list.append(roc_auc_score(df_test_y['sentiment'].values, y_proba))
        print("fold", fold, "training_time:", time1 - time0, "pred_time:", time2 - time1, "auc_score:",
              auc_score_list[-1])

    return auc_score_list


if __name__ == "__main__":
    print("\n\n\n============[ PROJECT 3  EVALUATION STARTS HERE !!]=================\n")
    auc_score_list = evaluate_proj3()
    print("\n\n\n============[ PROJECT 3 EVALUATION RESULTS]=================\n")

    print(auc_score_list)

    print("\n\n\n============[ PROJECT 3 EVALUATION FINISHED !!]=================\n")