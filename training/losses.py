import torch
import torch.nn.functional as F
from torch import Tensor


def cross_entropy_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits, labels)


def log_prob_for_labels(logits: Tensor, labels: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def kl_divergence_probs(reference_probs: Tensor, model_probs: Tensor) -> Tensor:
    return F.kl_div(
        model_probs.log(),
        reference_probs,
        reduction="batchmean",
        log_target=False,
    )


def retain_ft_loss(retain_logits: Tensor, retain_labels: Tensor) -> Tensor:
    return cross_entropy_loss(retain_logits, retain_labels)


def dpo_like_loss(
    retain_logits: Tensor,
    retain_labels: Tensor,
    forget_logits: Tensor,
    forget_labels: Tensor,
    reference_retain_logits: Tensor,
    reference_forget_logits: Tensor,
    beta: float,
) -> Tensor:
    retain_score = beta * (
        log_prob_for_labels(retain_logits, retain_labels)
        - log_prob_for_labels(reference_retain_logits, retain_labels)
    )
    forget_score = beta * (
        log_prob_for_labels(forget_logits, forget_labels)
        - log_prob_for_labels(reference_forget_logits, forget_labels)
    )
    return -F.logsigmoid(retain_score - forget_score).mean()


def rmu_loss(
    retain_logits: Tensor,
    retain_labels: Tensor,
    forget_logits: Tensor,
    reference_retain_logits: Tensor,
    num_classes: int,
) -> Tensor:
    retain_ce = cross_entropy_loss(retain_logits, retain_labels)

    reference_retain_probs = F.softmax(reference_retain_logits, dim=-1)
    model_retain_probs = F.softmax(retain_logits, dim=-1)
    retain_kl = kl_divergence_probs(reference_retain_probs, model_retain_probs)

    uniform_target = torch.full_like(forget_logits, 1.0 / num_classes)
    model_forget_probs = F.softmax(forget_logits, dim=-1)
    refusal_kl = kl_divergence_probs(uniform_target, model_forget_probs)

    return retain_ce + 0.5 * retain_kl + refusal_kl


def random_target_loss(
    retain_logits: Tensor,
    retain_labels: Tensor,
    forget_logits: Tensor,
    random_forget_labels: Tensor,
    gamma: float,
) -> Tensor:
    retain_ce = cross_entropy_loss(retain_logits, retain_labels)
    forget_ce = cross_entropy_loss(forget_logits, random_forget_labels)
    return retain_ce + gamma * forget_ce
