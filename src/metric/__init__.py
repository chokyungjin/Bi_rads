from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


def get_metrics_PRFA(num_classes, prefix, threshold=0.5):
    "return MetricCollection of Precision, Recall and F1score"
    metrics = MetricCollection(
        [
            Precision(
                num_classes=num_classes,
                average="micro",
                threshold=threshold,
            ),
            Recall(
                num_classes=num_classes,
                average="micro",
                threshold=threshold,
            ),
            F1Score(
                num_classes=num_classes,
                average="micro",
                threshold=threshold,
            ),
            Accuracy(
                num_classes=num_classes,
                average="micro",
                threshold=threshold,
            )
        ]
    )
    return metrics.clone(prefix=prefix)

def get_metric_ACC(num_classes, prefix, threshold=0.5):
    "return MetricCollection of ACC"
    metrics = MetricCollection(
        [
            Accuracy(
                num_classes=num_classes,
                average="micro",
                threshold=threshold,
            )
        ]
    )
    return metrics.clone(prefix=prefix)