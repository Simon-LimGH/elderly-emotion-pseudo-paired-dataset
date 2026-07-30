"""
build_pseudo_paired_dataset.py
==============================
Reproducibly builds the pseudo-paired (face + text) emotion dataset used in the
paper "Elderly Emotion Detection Using a Multimodal Approach".

Because no public dataset pairs an elderly face with its age, emotion AND text
from the same person, each labelled face is paired with a *real* GoEmotions
sentence that expresses the SAME emotion. Face and text therefore do NOT come
from the same person -- they share only an emotion label. The set is a
proof-of-concept and any text-only / fused score measured on it is optimistic.

Inputs  (committed to the repo, no images required):
    faces_manifest/{train,val,test}.csv
        columns: source, image_relpath, filename, emotion_label, emotion_name

Output:
    pseudo_paired/{train,val,test}.csv
        the manifest columns + a `text` column (empty for the Contempt class,
        which has no counterpart in the 7 Ekman text emotions).

Reproducibility: a fixed seed (42) and the ordered GoEmotions pools make the
pairing deterministic. Re-running against the same GoEmotions release regenerates
the committed CSVs. The GoEmotions corpus is downloaded automatically from the
Hugging Face Hub (internet required on first run).

Usage:
    pip install -r requirements.txt
    python build_pseudo_paired_dataset.py                 # uses ./faces_manifest -> ./pseudo_paired
    python build_pseudo_paired_dataset.py --seed 42 --manifest-dir faces_manifest --out-dir pseudo_paired
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

# ----------------------------------------------------------------------------- config
SEED = 42
NUM_CLASSES = 8
LABEL_TO_EMOTION = {0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear",
                    4: "Happiness", 5: "Neutral", 6: "Sadness", 7: "Surprise"}
EMOTION_NAMES = [LABEL_TO_EMOTION[i] for i in range(NUM_CLASSES)]

# The 28 GoEmotions labels, in their official order.
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]

# GoEmotions (27 + neutral) collapsed to the 7 Ekman emotions.
GO_TO_EKMAN = {
    "admiration": "joy", "amusement": "joy", "approval": "joy", "caring": "joy", "desire": "joy",
    "excitement": "joy", "gratitude": "joy", "joy": "joy", "love": "joy", "optimism": "joy",
    "pride": "joy", "relief": "joy",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "disappointment": "sadness", "embarrassment": "sadness", "grief": "sadness",
    "remorse": "sadness", "sadness": "sadness",
    "confusion": "surprise", "curiosity": "surprise", "realization": "surprise", "surprise": "surprise",
    "neutral": "neutral",
}

# Which of the 8 facial slots each Ekman emotion fills. There is no "Contempt"
# (slot 1) in the text data, so Contempt faces keep an empty text.
EKMAN_TO_FACE8 = {"anger": 0, "disgust": 2, "fear": 3, "joy": 4,
                  "neutral": 5, "sadness": 6, "surprise": 7}

# The GoEmotions split each face split draws its text from.
FACE_TO_GO_SPLIT = {"train": "train", "val": "validation", "test": "test"}

GO_DATASET = "google-research-datasets/go_emotions"
GO_CONFIG = "simplified"
COLUMNS = ["source", "image_relpath", "filename", "emotion_label", "emotion_name", "text"]


def build_pool(go_split):
    """Group single-label GoEmotions sentences into the 8 facial emotion slots."""
    pool = {i: [] for i in range(NUM_CLASSES)}
    for row in go_split:
        labs = row["labels"]
        if len(labs) != 1:                       # keep only unambiguous, single-label comments
            continue
        ekman = GO_TO_EKMAN[GOEMOTIONS_LABELS[labs[0]]]
        pool[EKMAN_TO_FACE8[ekman]].append(row["text"])
    return pool


def main():
    ap = argparse.ArgumentParser(description="Build the pseudo-paired face+text emotion dataset.")
    here = Path(__file__).resolve().parent
    ap.add_argument("--manifest-dir", default=str(here / "faces_manifest"),
                    help="folder with train/val/test.csv face manifests")
    ap.add_argument("--out-dir", default=str(here / "pseudo_paired"),
                    help="folder to write the paired train/val/test.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    manifest_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading {GO_DATASET} ({GO_CONFIG}) ...")
    go = load_dataset(GO_DATASET, GO_CONFIG)
    pools = {gs: build_pool(go[gs]) for gs in ["train", "validation", "test"]}
    for gs in pools:
        sizes = {EMOTION_NAMES[i]: len(pools[gs][i]) for i in range(NUM_CLASSES) if pools[gs][i]}
        print(f"  text pool [{gs}]: {sizes}")

    # ONE generator, splits processed in this fixed order -> deterministic pairing.
    rng = np.random.default_rng(args.seed)
    for split in ["train", "val", "test"]:
        man_path = manifest_dir / f"{split}.csv"
        if not man_path.exists():
            print(f"[{split}] {man_path} not found - skipping")
            continue

        man = pd.read_csv(man_path)
        labels = man["emotion_label"].astype(int).values
        texts = np.empty(len(man), dtype=object)
        texts[:] = ""

        pool = pools[FACE_TO_GO_SPLIT[split]]
        for lbl in range(NUM_CLASSES):
            idx = np.where(labels == lbl)[0]
            p = pool.get(lbl, [])
            if len(idx) == 0 or len(p) == 0:            # Contempt (or empty pool) -> empty text
                continue
            pick = rng.choice(len(p), size=len(idx), replace=(len(p) < len(idx)))
            for j, ci in zip(idx, pick):
                texts[j] = p[ci]

        out = man.copy()
        out["text"] = texts
        out = out.reindex(columns=COLUMNS)
        out_path = out_dir / f"{split}.csv"
        out.to_csv(out_path, index=False)
        n_empty = int((out["text"].fillna("") == "").sum())
        print(f"[{split}] wrote {len(out)} rows -> {out_path}  ({n_empty} empty texts / Contempt)")


if __name__ == "__main__":
    main()
