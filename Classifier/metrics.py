def metric():
    """"
    As suggested in the description of the dataset,
    we use AUROC as a metric for performance.

    References
    ----------
    https://glassboxmedicine.com/2019/02/23/measuring-performance-auc-auroc/
    """
    pass

def accuracy_score(y_true, y_pred):
    """Accuracy score.

    Parmeters
    ---------
    y_true (array[float]): 1-d array of targets
    y_predicted (array[float]): 1-d array of predictions
    state_labels (array[float]): 1-d array of state representations for labels

    Returns
    -------
    score (float): the fraction of correctly classified samples
    """
    score = y_true == y_pred
    return score.sum() / len(y_true)