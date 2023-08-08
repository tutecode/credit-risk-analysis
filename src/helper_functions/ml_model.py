import pickle

import category_encoders as ce
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from helper_functions import data_utils, evaluation


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


def model_logistic_regression(df, save_model=False):
    """
    Train a logistic regression model on the given DataFrame and perform hyperparameter tuning using grid search.

    Args:
    - df (pd.DataFrame): The input DataFrame containing the data for training and evaluation.
    - save_model (bool, optional): If True, save the trained model to a file. Default is False.

    Returns:
    - grid_search (GridSearchCV): The trained logistic regression model with the best hyperparameters.
    """

    # Split the data into training, validation, and test sets
    X_train, y_train, X_test, y_test, X_val, y_val = data_utils.get_feature(df)
    X_train_reshape, y_train_reshape = data_utils.resampling(X_train, y_train)

    # Define the hyperparameter grid for grid search
    param_grid = {"C": [0.99, 0.10, 0.11, 0.111]}  # best = 0.11

    # Initialize logistic regression model
    logistic_model = LogisticRegression(
        penalty="l2", solver="sag", multi_class="auto", max_iter=500
    )

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(logistic_model, param_grid, cv=5)
    grid_search.fit(X_train_reshape, y_train_reshape)

    # Print the best score and model score on the validation set
    print("Best Score for Logistic Regression: ", grid_search.best_score_)
    print("Model score for Logistic Regression: %.3f" % grid_search.score(X_val, y_val))
    print("\n")

    # Make predictions on the test set
    y_hat = grid_search.predict(X_test)

    # Evaluate the model's performance
    accuracy = evaluation.get_performance(y_hat, y_test)

    # Plot the ROC curve
    evaluation.plot_roc(grid_search, y_test, X_test)

    # Save the trained model to a file if requested
    if save_model:
        filename = "logistic_regression.pk"
        pickle.dump(grid_search, open(filename, "wb"))

    # Print the best score again before returning the model
    print("Best Score for Logistic Regression: ", grid_search.best_score_)

    return grid_search


def model_catboost_classifier(df, save_model=False):
    """
    Train a CatBoost classifier model on the given DataFrame and perform hyperparameter tuning using grid search.

    Args:
    - df (pd.DataFrame): The input DataFrame containing the data for training and evaluation.
    - save_model (bool, optional): If True, save the trained model to a file. Default is False.

    Returns:
    - grid_search (GridSearchCV): The trained CatBoost classifier model with the best hyperparameters.
    """

    # Split the data into training, validation, and test sets
    X_train, y_train, X_test, y_test, X_val, y_val = data_utils.get_feature(df)
    X_train_reshape, y_train_reshape = data_utils.resampling(X_train, y_train)

    # Define the hyperparameter grid for grid search
    param_grid = {"learning_rate": [0.01, 0.1, 0.2], "depth": [4, 6, 8]}

    # Initialize CatBoost classifier model
    catboost_model = CatBoostClassifier(
        iterations=500, random_seed=42, logging_level="Silent"
    )

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(catboost_model, param_grid, cv=5)
    grid_search.fit(X_train_reshape, y_train_reshape)

    # Print the best score and model score on the validation set
    print("Best Score for CatBoost Classifier: ", grid_search.best_score_)
    print("Model score for CatBoost Classifier: %.3f" % grid_search.score(X_val, y_val))
    print("\n")

    # Make predictions on the test set
    y_hat = grid_search.predict(X_test)

    # Evaluate the model's performance
    accuracy = evaluation.get_performance(y_hat, y_test)

    # Plot the ROC curve
    evaluation.plot_roc(grid_search, y_test, X_test)

    # Save the trained model to a file if requested
    if save_model:
        filename = "catboost_classifier.pk"
        pickle.dump(grid_search, open(filename, "wb"))

    # Print the best score again before returning the model
    print("Best Score for CatBoost Classifier: ", grid_search.best_score_)

    return grid_search


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate a regression model's performance using Mean Squared Error (MSE) and R-squared (R2) metrics.

    Args:
    - model: The regression model to be evaluated.
    - X_train (array-like or pd.DataFrame): Training features.
    - X_test (array-like or pd.DataFrame): Testing features.
    - y_train (array-like or pd.Series): Training target.
    - y_test (array-like or pd.Series): Testing target.

    Returns:
    - mse (float): Mean Squared Error (MSE) between predicted and actual target values.
    - r2 (float): R-squared (R2) coefficient indicating the model's goodness of fit.
    """
    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict target values on the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Calculate R-squared (R2) score
    r2 = r2_score(y_test, y_pred)

    return mse, r2


def basic_models(df):
    """
    Train and evaluate various basic regression models on the provided dataset.

    Args:
    - df (pd.DataFrame): DataFrame containing the dataset.

    Returns:
    - None
    """
    # Create a copy of the DataFrame
    df_cop = df.copy()

    # Split the dataset into training, testing, and validation sets
    X_train, y_train, X_test, y_test, X_val, y_val = data_utils.get_feature(df_cop)
    X_train_reshape, y_train_reshape = data_utils.resampling(X_train, y_train)

    # Linear Regression Model
    linear_model = LinearRegression()
    mse_linear, r2_linear = evaluate_model(
        linear_model, X_train_reshape, X_test, y_train_reshape, y_test
    )

    # Logistic Regression Model
    logistic_model = LogisticRegression(max_iter=500)
    mse_logistic, r2_logistic = evaluate_model(
        logistic_model, X_train_reshape, X_test, y_train_reshape, y_test
    )

    # KNN Model (for regression problems)
    knn_model = KNeighborsRegressor()
    knn_model.fit(X_train_reshape, y_train_reshape)
    y_pred_knn = knn_model.predict(X_test)
    mse_knn, r2_knn = mean_squared_error(y_test, y_pred_knn), r2_score(
        y_test, y_pred_knn
    )

    # Gaussian Naive Bayes
    gnb_model = GaussianNB()
    mse_gnb, r2_gnb = evaluate_model(
        gnb_model, X_train_reshape, X_test, y_train_reshape, y_test
    )

    # Multi Layer Perceptron
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100,), activation="relu", solver="adam", random_state=42
    )
    mlp_model.fit(X_train_reshape, y_train_reshape)
    y_pred_mlp = mlp_model.predict(X_test)
    mse_mlp, r2_mlp = mean_squared_error(y_test, y_pred_mlp), r2_score(
        y_test, y_pred_mlp
    )

    # CatBoost
    catboost_model = CatBoostRegressor(random_state=42, verbose=False)
    catboost_model.fit(X_train, y_train)
    y_pred_catboost = catboost_model.predict(X_test)
    mse_catboost, r2_catboost = mean_squared_error(y_test, y_pred_catboost), r2_score(
        y_test, y_pred_catboost
    )

    # Ridge Regression Model
    ridge_model = Ridge(alpha=1.0)
    mse_ridge, r2_ridge = evaluate_model(
        ridge_model, X_train_reshape, X_test, y_train_reshape, y_test
    )

    # LASSO Regression Model
    lasso_model = Lasso(alpha=1.0)
    mse_lasso, r2_lasso = evaluate_model(
        lasso_model, X_train_reshape, X_test, y_train_reshape, y_test
    )

    # Decision Tree Model
    decision_tree_model = DecisionTreeRegressor()
    mse_dt, r2_dt = evaluate_model(
        decision_tree_model, X_train_reshape, X_test, y_train_reshape, y_test
    )

    # Random Forest Classifier
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_reshape, y_train_reshape)
    y_pred_rf = rf_model.predict(X_test)
    mse_rf, r2_rf = mean_squared_error(y_test, y_pred_rf), r2_score(y_test, y_pred_rf)

    # Prepare and display the results in a DataFrame
    results = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Logistic Regression",
                "KNeighborsRegressor",
                "Gaussian Naive Bayes",
                "Multi Layer Perceptron",
                "CatBoost",
                "Ridge Regression",
                "LASSO Regression",
                "Decision Tree",
                "Random Forest",
            ],
            "MSE": [
                mse_linear,
                mse_logistic,
                mse_knn,
                mse_gnb,
                mse_mlp,
                mse_catboost,
                mse_ridge,
                mse_lasso,
                mse_dt,
                mse_rf,
            ],
            "R²": [
                r2_linear,
                r2_logistic,
                r2_knn,
                r2_gnb,
                r2_mlp,
                r2_catboost,
                r2_ridge,
                r2_lasso,
                r2_dt,
                r2_rf,
            ],
        }
    )

    # Display the results
    print(results)
