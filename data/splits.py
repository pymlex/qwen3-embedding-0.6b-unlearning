from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from constants import LABEL2ID
from schemas import Config


def load_reviews(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, sep="\t", quoting=1)
    frame = frame.rename(columns=str.strip)
    frame["label_id"] = frame["sentiment"].map(LABEL2ID)
    return frame


def split_by_class(
    frame: pd.DataFrame,
    test_per_class: int,
    valid_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts = []
    valid_parts = []
    test_parts = []

    for sentiment in LABEL2ID:
        class_frame = frame[frame["sentiment"] == sentiment].sample(
            frac=1.0,
            random_state=seed,
        )
        test_parts.append(class_frame.iloc[:test_per_class])
        valid_parts.append(class_frame.iloc[test_per_class : test_per_class + valid_per_class])
        train_parts.append(class_frame.iloc[test_per_class + valid_per_class :])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    valid_df = pd.concat(valid_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    return train_df, valid_df, test_df


def build_retain_forget(train_df: pd.DataFrame, forget_class: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    forget_df = train_df[train_df["sentiment"] == forget_class].reset_index(drop=True)
    retain_df = train_df[train_df["sentiment"] != forget_class].reset_index(drop=True)
    return retain_df, forget_df


def prepare_splits(config: Config) -> dict[str, pd.DataFrame]:
    frame = load_reviews(config.data.csv_path)
    train_df, valid_df, test_df = split_by_class(
        frame,
        config.data.test_per_class,
        config.data.valid_per_class,
        config.data.seed,
    )
    retain_train_df, forget_train_df = build_retain_forget(train_df, config.data.forget_class)

    retain_test_df = test_df[test_df["sentiment"] != config.data.forget_class].reset_index(drop=True)
    forget_test_df = test_df[test_df["sentiment"] == config.data.forget_class].reset_index(drop=True)

    config.paths.splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.paths.splits_dir / "train.csv", index=False)
    valid_df.to_csv(config.paths.splits_dir / "valid.csv", index=False)
    test_df.to_csv(config.paths.splits_dir / "test.csv", index=False)
    retain_train_df.to_csv(config.paths.splits_dir / "retain_train.csv", index=False)
    forget_train_df.to_csv(config.paths.splits_dir / "forget_train.csv", index=False)
    retain_test_df.to_csv(config.paths.splits_dir / "retain_test.csv", index=False)
    forget_test_df.to_csv(config.paths.splits_dir / "forget_test.csv", index=False)

    return {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
        "retain_train": retain_train_df,
        "forget_train": forget_train_df,
        "retain_test": retain_test_df,
        "forget_test": forget_test_df,
    }
