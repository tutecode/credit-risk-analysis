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
# Make use of settings.py module to get Redis settings like host, port, etc.
# Connect to Redis
db = redis.Redis(
    host=settings.REDIS_IP, 
    port=settings.REDIS_PORT, 
    db=settings.REDIS_DB_ID,
    #charset="utf-8",
    #decode_responses=True
)

def load_model(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            model_tmp = pickle.load(f)
            logger.info("model:",model_tmp)
    else:
        raise FileNotFoundError(f"The file {filename} does not exist.")

    return model_tmp.best_estimator_,model_tmp.best_score_


def predict(data):
    # change permission of upload folder
    # os.chmod(settings.UPLOAD_FOLDER, 0o777)
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    class_name = None
    pred_probability = None

    # path of model
    model_file_path = "logistic_regression.pk"
    logger.info("despues data:",data)
    # Load model
    model, model_score = load_model(model_file_path)
    x = np.array(data)
    logger.info(f"data is: {x}")
    logger.info(f"Size of data is: {x.shape}")
    # adapt to one sample
    x_sample = x.reshape(1, -1)
    
    # Convert the 1D array into a 2D array (matrix) and pass it to the model for prediction
    class_name = model.predict(x_sample)[0]
    logger.info(f"classname: {class_name}")
    pred_probability = model.predict_proba(x_sample)[0]
    logger.info(f"classname: {class_name}")
    # get idx to prob using the class 
    idx_prob = np.where(model.classes_==class_name)[0][0]
    pred_probability = pred_probability[idx_prob]
    logger.info(f"predict_proba: {pred_probability}")
    model_name = model.__class__.__name__
    model_score = round(model_score,2)
    logger.info("name: {}, score: {}".format(model_name,model_score))
    return model_name, model_score, class_name, pred_probability

def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
        # Hint: You should be able to successfully implement the communication
        #       code with Redis making use of functions `brpop()` and `set()`.
        # TODO
        try:
            # 1. Take a new job from Redis
            queue_name, msg = db.brpop(settings.REDIS_QUEUE)
            logger.info("msg: {}".format(msg))
            #print(f"Queue name: {queue_name}, msg: {msg}")

            # 2. Run ML model on the given data
            newmsg = json.loads(msg)
            # print(f"name_image: {newmsg['image_name']}")
            logger.info("antes data:",newmsg["data"])
            # 2.1. only need the filename image the image object is loaded by the upload folder
            model_name, model_score, class_name, pred_probability = predict(newmsg["data"])
            logger.info("clas_prob: {} {}".format(class_name, pred_probability))
            # 3. Store model prediction in a dict with the following shape
            res_dict = {
                "model_name": model_name,
                "model_score": model_score,
                "prediction": str(class_name),
                "score": round(np.float64(pred_probability),2),
            }
            
            # 4. Store the results on Redis using the original job ID as the key
            # so the API can match the results it gets to the original job sent
            res_id = newmsg["id"]
            logger.info("res: {} {}".format(res_id, res_dict))
            # Here, you can see we use `json.dumps` to
            # serialize our dict into a JSON formatted string.
            db.set(res_id, json.dumps(res_dict))
        except:
            raise SystemExit("ERROR: Results Not Stored")
            

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)
if __name__ == "__main__":
    # Now launch process
    logger.info("Launching ML service...")
    classify_process()