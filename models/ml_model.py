import pickle
import os
import logging
import numpy as np
import settings
import redis
import json
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Connect to Redis and assign to variable `db``
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)


def load_model(filename):
    """
    Load a trained machine learning model from a file.

    This function loads a pickled machine learning model from a given file.
    The model is expected to be a scikit-learn GridSearchCV object.

    Parameters:
    - filename (str): Path to the file containing the pickled model.

    Returns:
    - best_estimator (object): The best estimator (trained model) from GridSearchCV.
    - best_score (float): The best score achieved by the model during GridSearchCV.
    """
    # Check if the given filename exists in the filesystem
    if os.path.exists(filename):
        # Open the file in binary read mode
        with open(filename, "rb") as f:
            # Load the pickled model object from the file
            model_tmp = pickle.load(f)
            # logger.info("model:", model_tmp)
    else:
        # If the file does not exist, raise a FileNotFoundError
        raise FileNotFoundError(f"The file {filename} does not exist.")

    return model_tmp.best_estimator_, model_tmp.best_score_

import numpy as np

def predict(data):
    """
    Make predictions using a trained machine learning model.

    This function takes input data, loads a pre-trained model, and makes predictions
    using the loaded model.

    Parameters:
    - data (list or numpy.ndarray): Input data for prediction.

    Returns:
    - model_name (str): Name of the loaded machine learning model.
    - model_score (float): Best score achieved by the loaded model during training.
    - class_name (str): Predicted class name for the input data.
    - pred_probability (float): Predicted probability of the predicted class.
    """
    
    # Initialize variables for storing prediction results
    class_name = None
    pred_probability = None

    # Define the path of the trained model file
    model_file_path = "logistic_regression.pk"

    # Load the pre-trained model and its best score
    model, model_score = load_model(model_file_path)

    # Convert input data into a numpy array
    x = np.array(data)
    #logger.info(f"data is: {x}")
    #logger.info(f"Size of data is: {x.shape}")

    # Reshape the input data to match the model's expected format (one sample)
    x_sample = x.reshape(1, -1)

    # Make predictions using the loaded model
    class_name = model.predict(x_sample)[0]
    pred_probability = model.predict_proba(x_sample)[0]

    # Get the index of the predicted class in the probability array
    idx_prob = np.where(model.classes_ == class_name)[0][0]
    pred_probability = pred_probability[idx_prob]

    # Get the name of the loaded model's class and round the model score
    model_name = model.__class__.__name__
    model_score = round(model_score, 2)

    # Return the prediction results
    return model_name, model_score, class_name, pred_probability



def classify_process():
    """
    Continuously process incoming jobs from the Redis queue.

    This function listens for incoming jobs in the Redis queue, runs the ML model on the data,
    stores the model prediction in Redis, and then waits for the next job.

    The loop runs indefinitely, continuously processing jobs from the queue.

    Note: This function should be run in a separate thread or process.

    Returns:
    - None
    """

    while True:
        # Take a new job from Redis
        msg = db.brpop(settings.REDIS_QUEUE, settings.SERVER_SLEEP)

        if msg is not None:
            # Extract the message content from the returned tuple
            msg = msg[1]

            # Run ML model on the given data
            newmsg = json.loads(msg)
            model_name, model_score, class_name, pred_probability = predict(
                newmsg["data"]
            )

            # Store model prediction in a dictionary
            res_dict = {
                "model_name": model_name,
                "model_score": model_score,
                "prediction": str(class_name),
                "score": round(np.float64(pred_probability), 2),
            }

            # Store the results on Redis using the original job ID as the key
            res_id = newmsg["id"]
            db.set(res_id, json.dumps(res_dict))

        # Sleep for a bit before processing the next job
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    logger.info("Launching ML service...")
    classify_process()
