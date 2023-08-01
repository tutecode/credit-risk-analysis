import pickle
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.impute import SimpleImputer
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




# Function to change te repated column name
def repeated_name(df1, df2):
    metadata = df2

    meta_cols = metadata["Var_Title"].to_list()
    meta_cols[43] = "MATE_EDUCATION_LEVEL"

    # Set the new column to the train_data and test_data
    df1.columns = meta_cols
    return df1


# shows only numerical columns
def unique_numerical(df1, df2):
    print("{:<32}{:<15}{}\n".format("Feature Name", "UniqueCount", "RangeMeta"))
    number_field_names = df1.select_dtypes("number").columns.to_list()
    metadata = df2
    metadata_dic = {colname: idx for idx, colname in enumerate(df1.columns)}

    for number_field in number_field_names:
        # print(number_field.unique())
        print(
            "{:<32}{:<15}{}".format(
                number_field,
                len(df1[number_field].unique()),
                metadata.iloc[metadata_dic[number_field], 2],
            )
        )

    
# shows only non numerical columns
def unique_categorical(df1, df2):
    category_field_names = df1.select_dtypes(exclude="number").columns.to_list()
    metadata_dic = {colname: idx for idx, colname in enumerate(df1.columns)}
    print("{:<32}{:<15}{}\n".format("Feature Name", "UniqueCount", "RangeMeta"))
    for categorical_field in category_field_names:
        print(
            "{:<32}{:<15}{}".format(
                categorical_field,
                len(df1[categorical_field].unique()),
                df2.iloc[metadata_dic[categorical_field], 2],
            )
        )


# for columns with lots of outliers
def proc_outliers(df, field):
    # impute nans with mean value of column
    df[field].replace({np.nan: df[field].mean()}, inplace=True)


# function for normalizing data at once
def normalized_data(df):
    df_cop = df.copy()
    target_col = "TARGET_LABEL_BAD=1"

    # 'PAYMENT_DAY': category = ["1 - 15", "16 - 30"]
    df_cop['PAYMENT_DAY'] = np.where(df_cop['PAYMENT_DAY'] <= 14, "1_14", "15_30")


    # 'MARITAL_STATUS': category =  {1:'single', 2:'married', 3:'other'}
    df_cop['MARITAL_STATUS'] = np.where(df_cop['MARITAL_STATUS'] == 1, "single",
                np.where(df_cop['MARITAL_STATUS'] == 2, "married", "other"))


    # 'QUANT_DEPENDANTS': numerical changes = [0, 1, 2, + 3]
    df_cop.loc[df_cop['QUANT_DEPENDANTS'] > 3, 'QUANT_DEPENDANTS'] = 3
    # 'HAS_DEPENDANTS': categorical column = {0:False, >0:True}
    df_cop['HAS_DEPENDANTS'] = np.where(df_cop['QUANT_DEPENDANTS'] >= 1, True, False)
    df_cop['HAS_DEPENDANTS'] =  df_cop['HAS_DEPENDANTS'].astype('bool')

    # "RESIDENCE_TYPE": numerical changes = {1: 'owned', 2:'mortgage', 3:'rented', 4:'family', 5:'other'}
    imp_const_zero = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    df_cop["RESIDENCE_TYPE"] = imp_const_zero.fit_transform(df_cop[["RESIDENCE_TYPE"]]).ravel()
    # categorical changes
    # mapping = {1: "owned", 2: "mortgage", 3: "rented", 4: "family", 5: "other"}
    df_cop["HAS_RESIDENCE"] = np.where(df_cop["RESIDENCE_TYPE"] == 1, True, False)
    df_cop['HAS_RESIDENCE'] =  df_cop['HAS_RESIDENCE'].astype('bool')

    # "MONTHS_IN_RESIDENCE": category = ['0 - 6 months', '< 1 year', '+ 1 year']
    df_cop["MONTHS_IN_RESIDENCE"] = np.where(df_cop["MONTHS_IN_RESIDENCE"] <= 6, '0_6',
            np.where(df_cop["MONTHS_IN_RESIDENCE"] <= 12, '6_12', '>_12'))


    # "MONTHLY_INCOMES_TOT" and "OTHER_INCOMES" changed by "OTHER_INCOMES"
    # added to personal income in order to increase people who has less than minimal salary
    df_cop["MONTHLY_INCOMES_TOT"] = (df_cop["PERSONAL_MONTHLY_INCOME"] + df_cop["OTHER_INCOMES"])

    df_cop["MONTHLY_INCOMES_TOT"] = pd.cut(df_cop["MONTHLY_INCOMES_TOT"],
                bins=[0, 650, 1320, 3323, 8560, float('inf')],
                labels=['[0_650]', '[650_1320]', '[1320_3323]', '[3323_8560]', '[>8560]'],
                right=False)


    # 'HAS_CARDS' category, replaces all cards.
    list_cards = ["FLAG_VISA", "FLAG_MASTERCARD", "FLAG_DINERS", "FLAG_AMERICAN_EXPRESS", "FLAG_OTHER_CARDS"]
    df_cop['HAS_CARDS'] = np.where(df_cop[list_cards].any(axis=1), True, False)
    df_cop['HAS_CARDS'] =  df_cop['HAS_CARDS'].astype('bool')
    

    # "QUANT_BANKING_ACCOUNTS" and "QUANT_SPECIAL_BANKING_ACCOUNTS" changed by "HAS_BANKING_ACCOUNTS"
    # added to personal income in order to increase people who has less than minimal salary
    df_cop["QUANT_BANKING_ACCOUNTS"] = (df_cop["QUANT_BANKING_ACCOUNTS"] + df_cop["QUANT_SPECIAL_BANKING_ACCOUNTS"])    
    
    # 'HAS_BANKING_ACCOUNTS' category, replaces all accounts.
    df_cop["HAS_BANKING_ACCOUNTS"] = np.where(df_cop["QUANT_BANKING_ACCOUNTS"] == 0, False, True)
    df_cop['HAS_BANKING_ACCOUNTS'] =  df_cop['HAS_BANKING_ACCOUNTS'].astype('bool')


    # 'PERSONAL_ASSETS_VALUE': changed to 'HAS_PERSONAL_ASSETS' = N, Y
    df_cop['HAS_PERSONAL_ASSETS'] = np.where(df_cop['PERSONAL_ASSETS_VALUE'] > 0, True, False)
    df_cop['HAS_PERSONAL_ASSETS'] =  df_cop['HAS_PERSONAL_ASSETS'].astype('bool')
    

    # 'QUANT_CARS':changed to 'HAS_CARS' = N, Y
    df_cop['HAS_CARS'] = np.where(df_cop['QUANT_CARS'] == 0, False, True)
    df_cop["HAS_CARS"] = df_cop["HAS_CARS"].astype('bool')


    # "APPLICATION_SUBMISSION_TYPE": 0 values changed to "Carga"
    df_cop.loc[df_cop["APPLICATION_SUBMISSION_TYPE"] != "Web", "APPLICATION_SUBMISSION_TYPE"] = "Carga"

    
    # 'SEX': deleted unknown values, changed to categorical
    df_cop.drop(df_cop[(df_cop["SEX"] == "N")].index,inplace=True,)
    df_cop.drop(df_cop[(df_cop["SEX"] == " ")].index,inplace=True,)
    
    
    # 'AGE'
    bins = [0, 18, 25, 35, 45, 60, float('inf')]
    labels = ['<_18', '18_25', '26_35', '36_45', '46_60', '>_60']
    df_cop['AGE'] = pd.cut(df_cop['AGE'], bins=bins, labels=labels) 

    return (df_cop, target_col)

def categorical_columns(df):
    # change columns to category, except bool columns
    object_columns = [col for col in df.columns if df[col].dtype != 'bool']
    df[object_columns] = df[object_columns].astype('category')
    return df


def delete_columns(df):

    # delete columns with single values
    num_unique_values = df.nunique()
    columns_to_drop = num_unique_values[num_unique_values == 1].index
    df.drop(columns=columns_to_drop, inplace=True)

    # delete columns according to our criteria
    drop_columns=['ID_CLIENT', # index 
                'POSTAL_ADDRESS_TYPE', # not valid proportion
                # 'QUANT_DEPENDANTS',  # delete??
                # 'HAS_DEPENDANTS', # delete??
                'STATE_OF_BIRTH', # too many null values
                'CITY_OF_BIRTH', # too many values
                'NACIONALITY', # not valid proportion
                # RESIDENCIAL_STATE', # delete??
                'RESIDENCIAL_CITY', # too many unique values
                'RESIDENCIAL_BOROUGH', # too many unique values
                "RESIDENCIAL_PHONE_AREA_CODE", # too many unique values
                # 'FLAG_RESIDENCIAL_PHONE', # DELETE? if not chart
                "RESIDENCE_TYPE", # changed by HAS_RESIDENCE
                # "HAS_RESIDENCE", # DELETE?
                # "FLAG_EMAIL", # DELETE? if not chart
                "PERSONAL_MONTHLY_INCOME", # changed by 'MONTHLY_INCOMES_TOT'
                'OTHER_INCOMES', # changed by 'MONTHLY_INCOMES_TOT'
                'FLAG_VISA', # replaced by 'HAS_CARDS'
                'FLAG_MASTERCARD', # replaced by 'HAS_CARDS'
                'FLAG_DINERS', # replaced by 'HAS_CARDS'
                'FLAG_AMERICAN_EXPRESS', # replaced by 'HAS_CARDS'
                'FLAG_OTHER_CARDS', # replaced by 'HAS_CARDS'
                'QUANT_BANKING_ACCOUNTS', # replaced by 'HAS_BANKING_ACCOUNTS'
                'QUANT_SPECIAL_BANKING_ACCOUNTS', # replaced by 'HAS_BANKING_ACCOUNTS'
                'PERSONAL_ASSETS_VALUE', # replaced by 'HAS_PERSONAL_ASSETS'
                'QUANT_CARS', # replaced by 'HAS_CARS'
                'PROFESSIONAL_STATE', # more than 60% of empty values
                'PROFESSIONAL_CITY', # too many different values
                'PROFESSIONAL_BOROUGH', # too many different values
                'PROFESSIONAL_PHONE_AREA_CODE', # more than 60% of empty values
                'MONTHS_IN_THE_JOB', # more than 95% of 0 as a value
                'PROFESSION_CODE', # not enough information, over 7k null values
                'OCCUPATION_TYPE', # not enough information, over 7k null values
                'MATE_PROFESSION_CODE', # over 50% of empty values
                'MATE_EDUCATION_LEVEL', # over 60% of empty values
                # 'PRODUCT', delete?, 3 different values, 
                'RESIDENCIAL_ZIP_3', # too many unique values'
                'PROFESSIONAL_ZIP_3'] # too many unique values'
    
    list_not_find = []
    list_removed = []
    
    for outside_column in drop_columns:
        if(outside_column in df.columns):
            list_removed.append(outside_column)
            df.drop(columns = outside_column, axis=1, inplace=True)
        else:
            list_not_find.append(outside_column)


    print("Those columns were removed: \n",list_removed)
    print("\nThose columns were not found: \n",list_not_find)

    return df


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


# Neural Networks












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

    # # changes to avoid errors in lightgbm
    # X_train_reshape.columns = X_train_reshape.columns.str.replace('[^a-zA-Z0-9_]', '_')
    # y_train_reshape = y_train_reshape.columns.str.replace('[^a-zA-Z0-9_]', '_')
    # X_test.columns, y_test.columns = X_test.columns.str.replace('[^a-zA-Z0-9_]', '_'), y_test.columns.str.replace('[^a-zA-Z0-9_]', '_')
    # X_val.columns, y_val.columns = X_val.columns.str.replace('[^a-zA-Z0-9_]', '_'), X_val.columns.str.replace('[^a-zA-Z0-9_]', '_')
    
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
    
    # # LightGBM
    # lgbm_model = LGBMRegressor(random_state=42)
    # lgbm_model.fit(X_train_reshape, y_train_reshape)
    # y_pred_lgbm = lgbm_model.predict(X_test)
    # mse_lgbm, r2_lgbm = mean_squared_error(y_test, y_pred_lgbm), r2_score(y_test, y_pred_lgbm)

    # CatBoost
    catboost_model = CatBoostRegressor(random_state=42, verbose=False)
    catboost_model.fit(X_train, y_train)
    y_pred_catboost = catboost_model.predict(X_test)
    mse_catboost, r2_catboost = mean_squared_error(y_test, y_pred_catboost), r2_score(y_test, y_pred_catboost)

    # XGBoost
    # xgboost_model = XGBRegressor(random_state=42)
    # xgboost_model.fit(X_train, y_train)
    # y_pred_xgboost = xgboost_model.predict(X_test)
    # mse_xgboost, r2_xgboost = mean_squared_error(y_test, y_pred_xgboost), r2_score(y_test, y_pred_xgboost)
    # print('7')

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

    # results = pd.DataFrame({
    #     'Model': ['Linear Regression', 'Logistic Regression', 'KNeighborsRegressor', 'Gaussian Naive Bayes', 'Multi Layer Perceptron', 'LightGBM', 'CatBoost', 'XGBoost', 'Ridge Regression', 'LASSO Regression', 'Decission Tree', 'Random Forest'],
    #     'MSE': [mse_linear, mse_logistic, mse_knn, mse_gnb, mse_mlp, mse_lgbm, mse_catboost, mse_xgboost, mse_ridge, mse_lasso, mse_dt, mse_rf],
    #     'R²': [r2_linear, r2_logistic, r2_knn, r2_gnb, r2_mlp, r2_lgbm, r2_catboost, r2_xgboost, r2_ridge, r2_lasso, r2_dt, r2_rf]
    # })

    print(results)

