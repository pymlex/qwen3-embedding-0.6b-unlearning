import numpy as np
import torch
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from constants import GOLD_LABEL_TO_FULL, RETAIN_FULL_LABEL_IDS


def multiclass_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(matthews_corrcoef(y_true, y_pred))


def batch_predict(
    model: torch.nn.Module,
    texts: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        logits = model(batch_texts, device)
        batch_predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        predictions.append(batch_predictions)
    return np.concatenate(predictions)


def batch_predict_probs(
    model: torch.nn.Module,
    texts: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    probabilities = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_probs = model.predict_probs(batch_texts, device).detach().cpu().numpy()
        probabilities.append(batch_probs)
    return np.concatenate(probabilities, axis=0)


def retain_probs_from_three_class(model_probs: np.ndarray) -> np.ndarray:
    subset = model_probs[:, RETAIN_FULL_LABEL_IDS]
    return subset / subset.sum(axis=1, keepdims=True)


def gold_predictions_to_full(gold_predictions: np.ndarray) -> np.ndarray:
    return np.vectorize(GOLD_LABEL_TO_FULL.get)(gold_predictions)


def kl_divergence(probs_p: np.ndarray, probs_q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs_p, eps, 1.0)
    q = np.clip(probs_q, eps, 1.0)
    kl_values = np.sum(p * np.log(p / q), axis=-1)
    return float(np.mean(kl_values))


def agreement_rate(y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> float:
    return float(np.mean(y_pred_a == y_pred_b))


def forget_class_mcc(
    y_pred: np.ndarray,
    forget_class_id: int,
    num_classes: int,
) -> float:
    """MCC on forget split where all ground-truth labels are the forget class.

    Multiclass MCC is degenerate when y_true contains a single class. We compare
    the forget-class prediction rate to random guessing at 1 / K.
    """
    prediction_rate = float(np.mean(y_pred == forget_class_id))
    random_rate = 1.0 / num_classes
    if prediction_rate >= 1.0:
        return 1.0
    if prediction_rate <= 0.0:
        return -1.0
    if abs(prediction_rate - random_rate) < 1e-12:
        return 0.0
    numerator = prediction_rate - random_rate
    denominator = np.sqrt(
        prediction_rate * (1.0 - prediction_rate) * random_rate * (1.0 - random_rate)
    )
    return float(numerator / denominator)


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def evaluate_split(
    model: torch.nn.Module,
    texts: list[str],
    labels: list[int],
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    label_array = np.asarray(labels, dtype=np.int64)
    predictions = batch_predict(model, texts, device, batch_size)
    return {
        "mcc": multiclass_mcc(label_array, predictions),
        "predictions": predictions,
        "labels": label_array,
    }


def evaluate_unlearning_metrics(
    model: torch.nn.Module,
    gold_model: torch.nn.Module,
    retain_texts: list[str],
    retain_labels: list[int],
    forget_texts: list[str],
    forget_labels: list[int],
    device: torch.device,
    batch_size: int,
    forget_class_id: int,
    num_classes: int,
) -> dict[str, float]:
    retain_eval = evaluate_split(model, retain_texts, retain_labels, device, batch_size)
    forget_preds = batch_predict(model, forget_texts, device, batch_size)
    forget_mcc = forget_class_mcc(forget_preds, forget_class_id, num_classes)

    retain_probs_model = retain_probs_from_three_class(
        batch_predict_probs(model, retain_texts, device, batch_size)
    )
    forget_probs_model = retain_probs_from_three_class(
        batch_predict_probs(model, forget_texts, device, batch_size)
    )
    retain_probs_gold = batch_predict_probs(gold_model, retain_texts, device, batch_size)
    forget_probs_gold = batch_predict_probs(gold_model, forget_texts, device, batch_size)

    retain_pred_gold = gold_predictions_to_full(
        batch_predict(gold_model, retain_texts, device, batch_size)
    )
    forget_pred_gold = gold_predictions_to_full(
        batch_predict(gold_model, forget_texts, device, batch_size)
    )

    return {
        "model_retain_mcc": retain_eval["mcc"],
        "model_forget_mcc": forget_mcc,
        "gold_kl_retain": kl_divergence(retain_probs_gold, retain_probs_model),
        "gold_kl_forget": kl_divergence(forget_probs_gold, forget_probs_model),
        "gold_agree_retain": agreement_rate(retain_eval["predictions"], retain_pred_gold),
        "gold_agree_forget": agreement_rate(forget_preds, forget_pred_gold),
    }
