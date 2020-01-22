class FoldMetrics:
    def __init__(self, roc_curve, pr_curve, avg_prec):
        self.roc_curve = roc_curve
        self.pr_curve = pr_curve
        self.avg_prec = avg_prec
