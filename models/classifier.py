import torch
import torch.nn as nn
from pathlib import Path
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    row_indices = torch.arange(batch_size, device=last_hidden_states.device)
    return last_hidden_states[row_indices, sequence_lengths]


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.layers(embeddings)


class QwenEmbeddingClassifier(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_classes: int,
        hidden_dim: int,
        max_length: int,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.encoder = AutoModel.from_pretrained(model_id)
        self.max_length = max_length
        embedding_dim = self.encoder.config.hidden_size
        self.classifier = MLPHead(embedding_dim, hidden_dim, num_classes)

    def encode_texts(self, texts: list[str], device: torch.device) -> Tensor:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = self.encoder(**batch)
        embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        return embeddings.float()

    def forward(self, texts: list[str], device: torch.device) -> Tensor:
        embeddings = self.encode_texts(texts, device)
        return self.classifier(embeddings)

    def predict_probs(self, texts: list[str], device: torch.device) -> Tensor:
        logits = self.forward(texts, device)
        return torch.softmax(logits, dim=-1)

    def save_pretrained(self, save_dir: str) -> None:
        self.encoder.save_pretrained(f"{save_dir}/encoder")
        self.tokenizer.save_pretrained(f"{save_dir}/encoder")
        torch.save(self.classifier.state_dict(), f"{save_dir}/classifier.pt")

    def load_pretrained(
        cls,
        save_dir: str,
        model_id: str,
        num_classes: int,
        hidden_dim: int,
        max_length: int,
    ) -> "QwenEmbeddingClassifier":
        checkpoint_dir = Path(save_dir)
        encoder_dir = checkpoint_dir / "encoder"
        model = cls(model_id, num_classes, hidden_dim, max_length)
        model.encoder = AutoModel.from_pretrained(str(encoder_dir), local_files_only=True)
        model.tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir), padding_side="left", local_files_only=True)
        model.classifier.load_state_dict(torch.load(checkpoint_dir / "classifier.pt", map_location="cpu"))
        return model
