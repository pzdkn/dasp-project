from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(preds, out_label_ids):
    """
    computes accuracy, f1-macro, f1-micro
    :param preds: preditions
    :param out_label_ids: output label
    :return: dict containing above mentioned metrics
    """
    f1_macro = f1_score(out_label_ids, preds, average="macro")
    f1_micro = f1_score(out_label_ids, preds, average="micro")
    accuracy = accuracy_score(out_label_ids, preds)
    return {"accuracy": accuracy, "f1_macro": f1_macro, "f1_micro": f1_micro}
