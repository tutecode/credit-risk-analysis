import pickle
import numpy as np
import pandas as pd
import os
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression


# to encode dataframe from front-end
def encoding(df, get_dummies=False):

    print("ENCODING")
    print(df)

    cols_to_encode = df.columns
    print(cols_to_encode)
    
    if get_dummies:
        df_dummy = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=True)
        return(df_dummy)
    
    else:
        cols_to_encode.remove('RESIDENCIAL_STATE') # this column will have category encoder due to amount of unic values
        
        # one-hot encoder
        oh_encoder = OneHotEncoder(drop='first', sparse=False) 
        encoded_onehot = oh_encoder.fit_transform(df[cols_to_encode])
        encoded_onehot = pd.DataFrame(encoded_onehot, columns=oh_encoder.get_feature_names_out(input_features=cols_to_encode))

        # category encoder
        ce_encoder = ce.BinaryEncoder(cols=['RESIDENCIAL_STATE'])
        encoded_category = ce_encoder.fit_transform(df[['RESIDENCIAL_STATE']])

        # join both encoders and target into the same dataframe
        df_encoded = pd.concat([encoded_onehot, encoded_category], axis=1)
        return (df_encoded)


def predict_target(df):
    
    #df_predict = df_encoded.copy()
    #filename = 'logistic_regression.pk'
    #model_lr = pickle.load(open(filename, 'rb')) # to load model...
#
    #y_pred = model_lr.predict(df_encoded)
#
    ## df_predict['TARGET_LABEL_BAD=1'] = y_pred
   #
    ## Ahora, y_new_pred contiene las predicciones para las nuevas instancias
    #return y_pred
    print("PREDICTION")
    print(df)

    # Docker
    #model_file_path = '/app/helper_function/logistic_regression.pk'  # Replace with the correct file path
    
    # Local
    model_file_path = '.\helper_function\logistic_regression.pk'


    #current_directory = os.getcwd()
    #script_path = os.path.abspath(__file__)
    #print("Script Path:", script_path)
    #print("Current Working Directory:", current_directory)
    #directory_path = "/app/helper_function/"
#
    ## Get the list of all files in the directory
    #files_in_directory = os.listdir(directory_path)
#
    ## Print the names of all files in the directory
    #print("Files in the directory:", files_in_directory)
    # Get the list of all files in the script path directory
    #files_in_directory = os.listdir(script_path)
    # Print the names of all files in the directory
    #print("Files in the directory:", files_in_directory)
    if os.path.exists(model_file_path):
        model_lr = pickle.load(open(model_file_path, 'rb'))
        y_pred = model_lr.predict(df)
        df['TARGET_LABEL_BAD=1'] = y_pred
    else:
        raise FileNotFoundError(f"The file {model_file_path} does not exist.")
    
    return df
