LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {value: key for key, value in LABEL2ID.items()}

RETAIN_CLASSES = ("positive", "neutral")
