from __future__ import annotations

from crohnstracker_shared.constants import KVASIR_LABEL_MAP, TARGET_LABELS


def map_label(raw: str) -> str:
    if raw in KVASIR_LABEL_MAP:
        return KVASIR_LABEL_MAP[raw]
    # fuzzy contains
    low = raw.lower()
    for key, val in KVASIR_LABEL_MAP.items():
        if key.lower() in low:
            return val
    return "other_unmapped"


def is_target(label: str) -> bool:
    return label in TARGET_LABELS
