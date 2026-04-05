from __future__ import annotations

LABEL_TO_DISPLAY_NAME = {
    "left_hand": "Left hand",
    "right_hand": "Right hand",
    "feet": "Feet",
    "tongue": "Tongue",
}

LABEL_TO_COLOR = {
    "left_hand": "#1f77b4",
    "right_hand": "#d62728",
    "feet": "#2ca02c",
    "tongue": "#ff7f0e",
}

LABEL_TO_INT = {
    "left_hand": 1,
    "right_hand": 2,
    "feet": 3,
    "tongue": 4,
}

INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}

BNCI2014001_CHANNEL_NAMES = [
    "FZ",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "P1",
    "PZ",
    "P2",
    "POZ",
]

