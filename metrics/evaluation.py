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
) -> dict[str, float]:
    retain_eval = evaluate_split(model, retain_texts, retain_labels, device, batch_size)
    forget_eval = evaluate_split(model, forget_texts, forget_labels, device, batch_size)

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
        "model_forget_mcc": forget_eval["mcc"],
        "gold_kl_retain": kl_divergence(retain_probs_gold, retain_probs_model),
        "gold_kl_forget": kl_divergence(forget_probs_gold, forget_probs_model),
        "gold_agree_retain": agreement_rate(retain_eval["predictions"], retain_pred_gold),
        "gold_agree_forget": agreement_rate(forget_eval["predictions"], forget_pred_gold),
    }
