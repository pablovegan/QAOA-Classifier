# coding=UTF-8
#
# Main python script to run our classifier

from zipfile import ZipFile

import pandas as pd
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import svm

from quantumclassifier import QuantumClassifier


if __name__ == "__main__":
    # Extract creditcard.csv dataset
    with ZipFile("Data/creditcard.csv.zip", "r") as zipObj:
        zipObj.extractall()

    # DATASET
    # Load the dataset into a dataframe and make the transformations mentioned in the notebook
    credit_df = pd.read_csv("creditcard.csv")
    credit_df["Time"] = credit_df["Time"] / 3600  # convert to hours
    credit_df[["Amount"]] = (
        credit_df[["Amount"]] - credit_df[["Amount"]].mean()
    ) / credit_df[
        ["Amount"]
    ].std()  # normalize Amount
    # drop unnecessary columns
    drop_list = ["V8", "V14", "V15", "V20", "V22", "V23", "V24", "V25", "V27", "V28"]
    credit_df = credit_df.drop(drop_list, axis=1)
    # Separate features from labels
    feature_df = credit_df[credit_df.columns.drop("Class")]
    label_df = credit_df[["Class"]]
    # Save dataframes as arrays
    X = np.asarray(feature_df)  # shape (284807, 30)
    Y = np.asarray(label_df)  # shape (284807, 1)
    del feature_df
    del label_df

    # Split the dataset into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.9, random_state=4
    )
    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()
    del X
    del Y

    # TRAINING CLASSICAL CLASSIFIER
    print("-" * 30)
    print("Run the classical classifier")
    print("-" * 30)
    # Support Vector Machine Classifier
    clf = svm.SVC(kernel="rbf")
    clf.fit(X_train, Y_train.ravel())
    Yhat = clf.predict(X_test)
    # Metric
    print(f"The AUROC score for SVM classifier is {roc_auc_score(Y_test, Yhat)}")

    # TRAINING QUANTUM CLASSIFIER
    print("-" * 30)
    print("Run the quantum classifier")
    print("-" * 30)

    QClass = QuantumClassifier(X_train, Y_train, X_test, Y_train)
    QClass.optimize_circ()
