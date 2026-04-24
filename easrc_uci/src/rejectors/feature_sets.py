from __future__ import annotations

FEATURE_SETS: dict[str, list[str]] = {
    "ConfOnly": [
        "max_prob",
        "entropy",
        "margin",
        "energy",
    ],
    "XAIOnly": [
        "attr_entropy",
        "attr_stability",
        "topk_mass",
        "xai_unreliability",
    ],
    "BioOnlyProxy": [
        "proxy_alignment",
        "proxy_bio_unreliability",
        "random_proxy_alignment",
        "proxy_alignment_gap",
    ],
    "EASRCFullProxy": [
        "max_prob",
        "entropy",
        "margin",
        "energy",
        "attr_entropy",
        "attr_stability",
        "topk_mass",
        "xai_unreliability",
        "proxy_alignment",
        "proxy_bio_unreliability",
        "random_proxy_alignment",
        "proxy_alignment_gap",
    ],
}


LEARNED_REJECTOR_METHODS = [
    "ConfOnly",
    "XAIOnly",
    "BioOnlyProxy",
    "EASRCFullProxy",
]


SCORE_ONLY_METHODS = [
    "NoReject",
    "MaxProb",
    "Entropy",
    "Margin",
    "Energy",
]
