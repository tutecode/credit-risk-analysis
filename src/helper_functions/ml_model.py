import pickle
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from helper_functions import data_utils, evaluation

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# encode for model using get_dummies or onehot - category encoder
def encoding(df, get_dummies=False, target='TARGET_LABEL_BAD=1'):

    df_cop = df.drop(columns = target)
    cols_to_encode = [col for col in df.columns if col != target]
    
    if get_dummies:
        df_encoded = pd.get_dummies(data=df_cop, columns=cols_to_encode, drop_first=True)
        df_target = pd.DataFrame(df[target]).astype('uint8')
        df_dummy = pd.concat([df_encoded, df_target], axis=1) # join target to df
        return(df_dummy)
    
    else:
        cols_to_encode.remove('RESIDENCIAL_STATE') # this column will have category encoder due to amount of unic values
        
        # one-hot encoder
        oh_encoder = OneHotEncoder(drop='first', sparse=False) 
        encoded_onehot = oh_encoder.fit_transform(df_cop[cols_to_encode])
        encoded_onehot = pd.DataFrame(encoded_onehot, columns=oh_encoder.get_feature_names_out(input_features=cols_to_encode))

        # category encoder
        ce_encoder = ce.BinaryEncoder(cols=['RESIDENCIAL_STATE'])
        encoded_category = ce_encoder.fit_transform(df_cop[['RESIDENCIAL_STATE']])

        # join both encoders and target into the same dataframe
        df_target = pd.DataFrame(df[target]).astype('int64')
        df_encoded = pd.concat([encoded_onehot, encoded_category, df_target], axis=1)
        return (df_encoded)


# logistic regression model
def model_logistic_regression(df, save_model = False):
    
    X_train, y_train, X_test, y_test, X_val, y_val = data_utils.get_feature(df) 
    X_train_reshape, y_train_reshape = data_utils.resampling(X_train, y_train)

    param_grid = {'C': [0.99, 0.10, 0.11,  0.111]} # best = 0.11
    logistic_model = LogisticRegression(penalty='l2', solver='sag', multi_class='auto', max_iter=500)
    grid_search = GridSearchCV(logistic_model, param_grid, cv=5)
    grid_search.fit(X_train_reshape, y_train_reshape)
    
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
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


# basic models 
def basic_models (df):

    df_cop = df.copy()
    X_train, y_train, X_test, y_test, X_val, y_val = data_utils.get_feature(df_cop) 
    X_train_reshape, y_train_reshape = data_utils.resampling(X_train, y_train)

    # Linear Regression Model
    linear_model = LinearRegression()
    mse_linear, r2_linear = evaluate_model(linear_model, X_train_reshape, X_test, y_train_reshape, y_test)
    
    # Logistic Regression Model
    logistic_model = LogisticRegression(max_iter=500)
    mse_logistic, r2_logistic = evaluate_model(logistic_model, X_train_reshape, X_test, y_train_reshape, y_test)
    
    # KNN Model # for regression problems
    knn_model = KNeighborsRegressor()
    knn_model.fit(X_train_reshape, y_train_reshape)
    y_pred_knn = knn_model.predict(X_test)
    mse_knn, r2_knn = mean_squared_error(y_test, y_pred_knn), r2_score(y_test, y_pred_knn)
    
    # Gaussian Naive Bayes
    gnb_model = GaussianNB()
    mse_gnb, r2_gnb = evaluate_model(gnb_model, X_train_reshape, X_test, y_train_reshape, y_test)
    
    # Multi Layer Perceptron
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)
    mlp_model.fit(X_train_reshape, y_train_reshape)
    y_pred_mlp = mlp_model.predict(X_test)
    mse_mlp, r2_mlp = mean_squared_error(y_test, y_pred_mlp), r2_score(y_test, y_pred_mlp)
   
    # CatBoost
    catboost_model = CatBoostRegressor(random_state=42, verbose=False)
    catboost_model.fit(X_train, y_train)
    y_pred_catboost = catboost_model.predict(X_test)
    mse_catboost, r2_catboost = mean_squared_error(y_test, y_pred_catboost), r2_score(y_test, y_pred_catboost)

    # Ridge Regression Model
    ridge_model = Ridge(alpha=1.0)
    mse_ridge, r2_ridge = evaluate_model(ridge_model, X_train_reshape, X_test, y_train_reshape, y_test)

    # LASSO Regression Model
    lasso_model = Lasso(alpha=1.0)
    mse_lasso, r2_lasso = evaluate_model(lasso_model, X_train_reshape, X_test, y_train_reshape, y_test)

    # Decission Tree Model
    decision_tree_model = DecisionTreeRegressor()
    mse_dt, r2_dt = evaluate_model(decision_tree_model, X_train_reshape, X_test, y_train_reshape, y_test)

    # Random Forest Classifier
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_reshape, y_train_reshape)
    y_pred_rf = rf_model.predict(X_test)
    mse_rf, r2_rf = mean_squared_error(y_test, y_pred_rf), r2_score(y_test, y_pred_rf)

    results = pd.DataFrame({
        'Model': ['Linear Regression', 'Logistic Regression', 'KNeighborsRegressor', 'Gaussian Naive Bayes', 'Multi Layer Perceptron',  'CatBoost', 'Ridge Regression', 'LASSO Regression', 'Decission Tree', 'Random Forest'],
        'MSE': [mse_linear, mse_logistic, mse_knn, mse_gnb, mse_mlp, mse_catboost, mse_ridge, mse_lasso, mse_dt, mse_rf],
        'R²': [r2_linear, r2_logistic, r2_knn, r2_gnb, r2_mlp, r2_catboost, r2_ridge, r2_lasso, r2_dt, r2_rf]
    })


    print(results)
