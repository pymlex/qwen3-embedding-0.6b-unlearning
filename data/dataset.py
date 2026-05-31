import pandas as pd
import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.reviews = frame["review"].astype(str).tolist()
        self.labels = frame["label_id"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "text": self.reviews[index],
            "labels": self.labels[index],
        }


class PairDataset(Dataset):
    def __init__(self, retain_frame: pd.DataFrame, forget_frame: pd.DataFrame):
        self.retain_reviews = retain_frame["review"].astype(str).tolist()
        self.retain_labels = retain_frame["label_id"].astype(int).tolist()
        self.forget_reviews = forget_frame["review"].astype(str).tolist()
        self.forget_labels = forget_frame["label_id"].astype(int).tolist()
        self.length = max(len(self.retain_reviews), len(self.forget_reviews))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        retain_index = index % len(self.retain_reviews)
        forget_index = index % len(self.forget_reviews)
        return {
            "retain_text": self.retain_reviews[retain_index],
            "retain_label": self.retain_labels[retain_index],
            "forget_text": self.forget_reviews[forget_index],
            "forget_label": self.forget_labels[forget_index],
        }


def collate_text_batch(batch: list[dict[str, object]], text_key: str, label_key: str) -> dict[str, list]:
    return {
        "texts": [item[text_key] for item in batch],
        "labels": torch.tensor([item[label_key] for item in batch], dtype=torch.long),
    }


def collate_review_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return collate_text_batch(batch, "text", "labels")


def collate_pair_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    retain = collate_text_batch(batch, "retain_text", "retain_label")
    forget = collate_text_batch(batch, "forget_text", "forget_label")
    return {
        "retain_texts": retain["texts"],
        "retain_labels": retain["labels"],
        "forget_texts": forget["texts"],
        "forget_labels": forget["labels"],
    }
