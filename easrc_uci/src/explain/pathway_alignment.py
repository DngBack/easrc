from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def load_gmt(path: str | Path) -> dict[str, set[str]]:
    """
    Load MSigDB-style GMT: each line is name\\tdescription\\tgene1\\tgene2\\t...
    (description may be empty). Gene symbols are uppercased for matching.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GMT file not found: {path}")

    pathways: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if name:
                pathways[name] = genes
    if not pathways:
        raise ValueError(f"No pathways parsed from GMT: {path}")
    return pathways


def feature_name_to_index(feature_names: list[str]) -> dict[str, int]:
    """Map uppercased feature names to column indices (first occurrence wins)."""
    out: dict[str, int] = {}
    for i, f in enumerate(feature_names):
        key = str(f).strip().upper()
        if key and key not in out:
            out[key] = i
    return out


def indices_for_genes(
    genes: Iterable[str],
    gene_to_idx: dict[str, int],
) -> np.ndarray:
    idx = [gene_to_idx[g.upper()] for g in genes if str(g).strip().upper() in gene_to_idx]
    return np.array(sorted(set(idx)), dtype=int)


def build_class_pathway_groups(
    class_names: list[str],
    class_pathways: dict[str, list[str]],
    pathway_defs: dict[str, set[str]],
    gene_to_idx: dict[str, int],
) -> dict[int, np.ndarray]:
    """
    For each class index c, union all genes from listed pathway IDs (GMT names),
    then return feature indices present in X.
    """
    groups: dict[int, np.ndarray] = {}
    for c, label in enumerate(class_names):
        pathway_ids = class_pathways.get(label, [])
        gene_union: set[str] = set()
        for pid in pathway_ids:
            gene_union |= pathway_defs.get(pid, set())
        groups[c] = indices_for_genes(gene_union, gene_to_idx)
    return groups


def attribution_mass_in_predicted_pathway(
    attributions: np.ndarray,
    predicted_classes: np.ndarray,
    groups: dict[int, np.ndarray],
    eps: float = 1e-12,
) -> np.ndarray:
    """Same contract as proxy_bio.attribution_mass_in_predicted_group: deployable (uses y_pred)."""
    abs_attr = np.abs(attributions)
    total_mass = abs_attr.sum(axis=1) + eps
    align = np.zeros(attributions.shape[0], dtype=np.float32)
    for i, pred in enumerate(predicted_classes):
        g = groups[int(pred)]
        if g.size == 0:
            align[i] = 0.0
        else:
            align[i] = abs_attr[i, g].sum() / total_mass[i]
    return np.clip(align, 0.0, 1.0)


def make_random_groups_matched_size(
    num_classes: int,
    num_features: int,
    target_sizes: dict[int, int],
    n_random_groups: int = 10,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """
    Random gene index sets per class, same cardinality as pathway group for that class.
    Shape per class: [n_random_groups, group_size_c]
    """
    rng = np.random.default_rng(seed)
    random_groups: dict[int, np.ndarray] = {}
    for c in range(num_classes):
        k = min(int(target_sizes.get(c, 0)), num_features)
        if k <= 0:
            random_groups[c] = np.zeros((n_random_groups, 0), dtype=int)
            continue
        rows = []
        for _ in range(n_random_groups):
            pick = rng.choice(num_features, size=k, replace=False)
            rows.append(np.sort(pick))
        random_groups[c] = np.stack(rows, axis=0)
    return random_groups


def random_pathway_control_alignment(
    attributions: np.ndarray,
    predicted_classes: np.ndarray,
    random_groups: dict[int, np.ndarray],
    eps: float = 1e-12,
) -> np.ndarray:
    """Mean mass in random same-size gene sets (negative control)."""
    abs_attr = np.abs(attributions)
    total_mass = abs_attr.sum(axis=1) + eps
    align = np.zeros(attributions.shape[0], dtype=np.float32)
    for i, pred in enumerate(predicted_classes):
        groups_c = random_groups[int(pred)]
        if groups_c.size == 0:
            align[i] = 0.0
            continue
        masses = []
        for group in groups_c:
            if group.size == 0:
                masses.append(0.0)
            else:
                masses.append(abs_attr[i, group].sum() / total_mass[i])
        align[i] = float(np.mean(masses))
    return np.clip(align, 0.0, 1.0)


def pathway_groups_to_jsonable(
    groups: dict[int, np.ndarray],
    class_names: list[str],
    feature_names: list[str],
) -> dict:
    """For saving pathway_groups.json: class label -> list of gene symbols."""
    out: dict = {}
    for c, idx in groups.items():
        label = class_names[c] if c < len(class_names) else str(c)
        genes = [feature_names[i] for i in idx.tolist()]
        out[str(label)] = {"class_index": c, "gene_indices": idx.astype(int).tolist(), "genes": genes}
    return out
