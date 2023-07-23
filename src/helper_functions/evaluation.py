from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.preprocessing import label_binarize
import seaborn as sns

def print_performance(accuracy, precision, recall, f1_score,report):
    # Print metrics, don't change this code!
    print("Model Performance metrics:")
    print("-" * 30)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("\nModel Classification report:")
    print("-" * 30)
    print(report)

def get_performance(
    predictions: Union[List, np.ndarray],
    y_test: Union[List, np.ndarray],
    labels: Optional[Union[List, np.ndarray]] = [0, 1],
    verbose: bool=False,
) -> Tuple[float, float, float, float]:
    """
    Get model performance using different metrics.

    Args:
        predictions : Union[List, np.ndarray]
            Predicted labels, as returned by a classifier.
        y_test : Union[List, np.ndarray]
            Ground truth (correct) labels.
        labels : Union[List, np.ndarray]
            Optional display names matching the labels (same order).
            Used in `classification_report()`.

    Return:
        accuracy : float
        precision : float
        recall : float
        f1_score : float
    """
    # Compute metrics
    # Use sklearn.metrics.accuracy_score
    accuracy = metrics.accuracy_score(y_test, predictions)
    # Use sklearn.metrics.precision_score
    precision = metrics.precision_score(y_test, predictions,pos_label=0)
    # Use sklearn.metrics.recall_score
    recall = metrics.recall_score(y_test, predictions,pos_label=0)
    # Use sklearn.metrics.f1_score
    f1_score = metrics.f1_score(y_test, predictions,pos_label=0)
    # Use sklearn.metrics.classification_report
    report = metrics.classification_report(y_test, predictions, labels=labels)

    return accuracy, precision, recall, f1_score, report


def plot_metrics(
    model: BaseEstimator, 
    y_test: Union[List, np.ndarray], 
    features: np.ndarray,
    predictions: Union[List, np.ndarray],
    labels: Optional[Union[List, np.ndarray]] = [0, 1]
):
    """
    Plot metric matrix confussion and roc curve.

    Args:
        model : BaseEstimator
            Classifier model.
        y_test : Union[List, np.ndarray]
            Ground truth (correct) labels.
        features : List[int]
            Dataset features used to evaluate the model.
        predictions: List[int]
            Dataset of predictions
        labels: List[int]
            Labels for matrix confussion

    Return:
        roc_auc : float
            ROC AUC Score.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("Metrics of " + model.__class__.__name__)
    fig.align_labels()

    #Confusion Matrix, use sklearn.metrics.confusion_matrix
    cm = metrics.confusion_matrix(y_test, predictions, labels=labels)

    # Change figure size and increase dpi for better resolution
    #sns.set(ax=axes[1],font_scale = 0.6)
    sns.heatmap(cm, annot=True,fmt='d',vmin=0,ax=axes[0])
    axes[0].set_xlabel("Predicted Data", fontsize=6, labelpad=20)
    axes[0].xaxis.set_ticklabels(["Aproved Credit", "Rejected Credit"])
    axes[0].set_ylabel("Actual Data", fontsize=6, labelpad=20)
    axes[0].yaxis.set_ticklabels(["Aproved Credit", "Rejected Credit"])
    axes[0].set_title("Confusion Matrix for Credit Risk", fontsize=14, pad=20)
    
    #variables for roc curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    class_labels = model.classes_
    y_test = label_binarize(y_test,classes=class_labels)

    prob = model.predict_proba(features)
    y_score = prob[:, prob.shape[1] - 1]

    # compute roc curve with y_test and y_score (probs)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
    # compute area under curve between fpr and tpr
    roc_auc = metrics.auc(fpr, tpr)

    axes[1].plot(fpr, tpr, label=f"ROC curve (area = {roc_auc})", linewidth=2.5)
    axes[1].plot([0, 1], [0, 1], "k--", label=f"boundary")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("Receiver Operating Characteristic (ROC) Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()
    plt.close(fig)
