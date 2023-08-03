import pickle
import os


def predict_target(df):
    # Docker
    model_file_path = "/app/helper_function/logistic_regression.pk"

    # Local
    # model_file_path = '.\helper_function\logistic_regression.pk'

    if os.path.exists(model_file_path):
        # Load model
        model_lr = pickle.load(open(model_file_path, "rb"))

        # Replace NaN values with 0.0
        df = df.fillna(0.0)

        # Extract the first row from the DataFrame as a 1D array (a list)
        df_row = df.iloc[0].values.tolist()

        # Convert the 1D array into a 2D array (matrix) and pass it to the model for prediction
        y_prob = model_lr.predict_proba([df_row])

        # Get the probability of class 1 (approved)
        prob_class_1 = y_prob[0][1]

        # Set a threshold (e.g., 0.5) to convert probabilities to binary predictions
        if prob_class_1 > 0.5:
            prediction = 1  # Approved (class 1)
        else:
            prediction = 0  # Rejected (class 0)

    else:
        raise FileNotFoundError(f"The file {model_file_path} does not exist.")

    return prediction
