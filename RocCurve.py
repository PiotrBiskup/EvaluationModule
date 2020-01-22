from sklearn import metrics


class RocCurve:
    def __init__(self, fpr, tpr):
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = metrics.auc(fpr, tpr)
