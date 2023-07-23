from helper_functions import evaluation

def model_fit_predict(classifier,feature_target):
    # logistic regression
    X_train = feature_target["X_train"]
    y_train = feature_target["y_train"]
    X_val = feature_target["X_val"]
    y_val = feature_target["y_val"]
    X_test = feature_target["X_test"]
    y_test = feature_target["y_test"]

    classifier.fit(feature_target["X_train"], feature_target["y_train"].ravel())
    classifier_predictions = classifier.predict(feature_target["X_test"])
    return classifier_predictions