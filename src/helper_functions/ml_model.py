import pickle
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from helper_functions import data_utils, evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, RidgeClassifier, Lasso
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report,auc,roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def encoding(df, get_dummies=False, target="TARGET_LABEL_BAD=1"):
    """
    Encode categorical features in a DataFrame using one-hot encoding or a combination of one-hot encoding and binary encoding.

    Args:
    - df (pd.DataFrame): The input DataFrame containing the data to be encoded.
    - get_dummies (bool, optional): If True, perform one-hot encoding using pandas' get_dummies method. If False, use one-hot encoding along with binary encoding.
    - target (str, optional): The name of the target column in the DataFrame. Default is "TARGET_LABEL_BAD=1".

    Returns:
    - df_encoded (pd.DataFrame): The encoded DataFrame with categorical features replaced by their encoded versions.
    """

    df_cop = df.drop(columns=target)
    cols_to_encode = [col for col in df.columns if col != target]

    if get_dummies:
        # Perform one-hot encoding using pandas' get_dummies
        df_encoded = pd.get_dummies(
            data=df_cop, columns=cols_to_encode, drop_first=True
        )
        df_target = pd.DataFrame(df[target]).astype("uint8")
        df_dummy = pd.concat([df_encoded, df_target], axis=1)  # Join target to df
        return df_dummy

    else:
        cols_to_encode.remove(
            "RESIDENCIAL_STATE"
        )  # Remove column that will be encoded using category encoder

        # Perform one-hot encoding
        oh_encoder = OneHotEncoder(drop="first", sparse=False)
        encoded_onehot = oh_encoder.fit_transform(df_cop[cols_to_encode])
        encoded_onehot = pd.DataFrame(
            encoded_onehot,
            columns=oh_encoder.get_feature_names_out(input_features=cols_to_encode),
        )

        # Perform binary encoding for a specific column
        ce_encoder = ce.BinaryEncoder(cols=["RESIDENCIAL_STATE"])
        encoded_category = ce_encoder.fit_transform(df_cop[["RESIDENCIAL_STATE"]])

        # Join both encoders and the target into the same dataframe
        df_target = pd.DataFrame(df[target]).astype("int64")
        df_encoded = pd.concat([encoded_onehot, encoded_category, df_target], axis=1)

        return df_encoded


# logistic regression model
def model_logistic_regression(df, save_model = False):
    
    X_train, y_train, X_test, y_test, X_val, y_val = data_utils.get_feature(df) 
    #X_train_reshape, y_train_reshape = data_utils.resampling(X_train, y_train)

    param_grid = {'C': [0.99, 0.10, 0.11,  0.111], "class_weight": [{0:1,1:2}]} # best = 0.11
    logistic_model = LogisticRegression(max_iter=200)
    grid_search = GridSearchCV(logistic_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    print("Best Score for Logistic Regression: ", grid_search.best_score_)
    print("model score for Logistic Regression: %.3f" % grid_search.score(X_val, y_val))
    print("\n")
    y_hat = grid_search.predict(X_test)

    accuracy = evaluation.get_performance(y_hat, y_test)
    evaluation.plot_roc(grid_search, y_test, X_test)
    if save_model:
        filename = 'logistic_regression.pk'
        pickle.dump(grid_search, open(filename, 'wb'))
        # rf = pickle.load(open(filename, 'rb')) # to load model...

    print("Best Score for Logistic Regression: ", grid_search.best_score_)

    return grid_search

# catboost classifier model
def model_catboost_classifier(df, save_model=False):

    X_train, y_train, X_test, y_test, X_val, y_val = data_utils.get_feature(df) 
    X_train_reshape, y_train_reshape = data_utils.resampling(X_train, y_train)

    param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'depth': [4, 6, 8]}
    catboost_model = CatBoostClassifier(iterations=500, random_seed=42, logging_level='Silent')
    grid_search = GridSearchCV(catboost_model, param_grid, cv=5)
    grid_search.fit(X_train_reshape, y_train_reshape)
    
    print("Best Score for CatBoost Classifier: ", grid_search.best_score_)
    print("Model score for CatBoost Classifier: %.3f" % grid_search.score(X_val, y_val))
    print("\n")
    y_hat = grid_search.predict(X_test)

    accuracy = evaluation.get_performance(y_hat, y_test)
    evaluation.plot_roc(grid_search, y_test, X_test)
    
    if save_model:
        filename = 'catboost_classifier.pk'
        pickle.dump(grid_search, open(filename, 'wb'))
        # catboost = pickle.load(open(filename, 'rb')) # to load model...
    

    print("Best Score for CatBoost Classifier: ", grid_search.best_score_)

    return grid_search


# evaluate model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,pos_label=0,average="binary")
    recall = recall_score(y_test, y_pred,pos_label=0,average="binary") 
    f1 = f1_score(y_test, y_pred,pos_label=0,average="binary")
    report = classification_report(y_test, y_pred, labels=[0, 1])
    # Print metrics, don't change this code!
    print("Model Performance metrics:")
    print("-" * 30)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nModel Classification report:")
    print("-" * 30)
    print(report)

    prob = model.predict_proba(X_test)
    y_score = prob[:, prob.shape[1] - 1]
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=0)
    roc = auc(fpr, tpr)
    #roc = roc_auc_score(y_test, y_pred)

    return accuracy, precision, recall, f1, roc, y_pred


# basic models 
def basic_models(df,save_model=False):
    X_train, y_train, X_test, y_test, X_val, y_val = data_utils.get_feature(df) 
    #X_train_reshape, y_train_reshape = data_utils.resampling(X_train, y_train)

    class_weights = {0:1,1:2}

    models = {
        'logistic_model': LogisticRegression(C=0.1,max_iter=200,class_weight=class_weights),
        'gnb_model': GaussianNB(),
        'catboost_model': CatBoostClassifier(random_state=42, verbose=False,class_weights=class_weights),
        #'ridge_model': RidgeClassifier(alpha=1.0,class_weight=class_weights),
        'decision_tree_model': DecisionTreeClassifier(class_weight=class_weights),
        'rf_model': RandomForestClassifier(n_estimators=100, random_state=42,class_weight=class_weights),
    }

    results = []

    for model_name, model in models.items():
        model_accuracy, model_precision, model_recall, model_f1, model_roc,y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

        results.append({
            'Model': model_name,
            'Accuracy': model_accuracy,
            'Precision': model_precision,
            'Recall': model_recall,
            'F-1': model_f1,
            'ROC': model_roc,
            "model_class": model,
        })


        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the confusion matrix using seaborn
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

        # Customize the plot
        classes = np.unique(y_test)
        ax.set_xticklabels(classes, rotation=0)
        ax.set_yticklabels(classes, rotation=0)
        ax.set_title('Confusion Matrix '+model_name)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        # Show the plot
        plt.show()

    results_df = pd.DataFrame(results)
    sorted_results = results_df.sort_values(by=["Precision"],ascending=False)
    best_class= sorted_results.loc[0,"model_class"]

    # save best_class
    if save_model:
        filename = '../../models/logistic_regression.pk'
        pickle.dump(best_class, open(filename, 'wb'))




