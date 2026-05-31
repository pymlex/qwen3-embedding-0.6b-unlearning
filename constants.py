LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {value: key for key, value in LABEL2ID.items()}

RETAIN_CLASSES = ("negative", "positive")

RETAIN_LABEL2ID = {
    "negative": 0,
    "positive": 1,
}

GOLD_LABEL_TO_FULL = {
    0: LABEL2ID["negative"],
    1: LABEL2ID["positive"],
}

RETAIN_FULL_LABEL_IDS = [LABEL2ID[label] for label in RETAIN_CLASSES]
