---
license: gpl-3.0
base_model: Qwen/Qwen3-Embedding-0.6B
tags:
  - machine-unlearning
  - text-classification
  - sentiment-analysis
  - russian
---

# Qwen3-Embedding-0.6B Unlearning Checkpoints

Fine-tuned and unlearned variants of [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) for three-class Russian sentiment classification on women's clothing reviews. Training code, dataset splits, metrics, and reproduction commands live in the GitHub repository:

https://github.com/pymlex/qwen3-embedding-0.6b-unlearning

## Checkpoints

| Folder | Description |
| --- | --- |
| `gold/` | Two-epoch full-data reference model |
| `original/` | Initialization copy for unlearning |
| `unlearn/retain_ft/` | Retain fine-tuning |
| `unlearn/dpo_like/` | DPO-like unlearning |
| `unlearn/rmu/` | RMU with uniform refusal target |
| `unlearn/random_target/` | Random target mislabelling on forget set |

Each checkpoint stores a fine-tuned encoder directory and `classifier.pt` MLP head weights.

## Metrics

Final numbers are produced by `python main.py evaluate` in the GitHub project. Multiclass Matthews Correlation Coefficient is reported on retain and forget test partitions together with KL divergence and prediction agreement against the gold model.

| Model | test MCC | model_retain_mcc | model_forget_mcc | gold_kl_retain | gold_kl_forget |
| --- | --- | --- | --- | --- | --- |
| gold | pending | pending | pending | pending | pending |
| original | pending | pending | pending | pending | pending |
| best unlearn | pending | pending | pending | pending | pending |

## Loading

```python
from models.classifier import QwenEmbeddingClassifier

model = QwenEmbeddingClassifier.load_pretrained(
    "pymlex/qwen3-embedding-0.6b-unlearning",
    model_id="Qwen/Qwen3-Embedding-0.6B",
    num_classes=3,
    hidden_dim=512,
    max_length=256,
)
```

Replace the first argument with a local path to `gold/`, `original/`, or any `unlearn/{method}/` folder after download.

The project is under GPL-3.0 license.
