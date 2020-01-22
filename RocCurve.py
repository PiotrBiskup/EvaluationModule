from sklearn import metrics


class RocCurve:
    def __init__(self, fpr, tpr, thresholds):
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds
        self.roc_auc = metrics.auc(fpr, tpr)
