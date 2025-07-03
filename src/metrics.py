from sklearn.metrics import _regression


def regression_report(y_true, y_pred):
    report = {}
    metrics = _regression.__ALL__
    for name in metrics:
        try:
            metric = getattr(_regression, name)(y_true, y_pred)
            report[name] = metric
        except Exception:
            pass
    return report
