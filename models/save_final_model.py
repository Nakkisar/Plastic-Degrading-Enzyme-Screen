"""
=====================================================================
 TRAIN AND SAVE A FINAL, DEPLOYABLE MODEL FOR ONE PLASTIC TYPE
=====================================================================
Cross-validation (as done in train_model.py) is for MEASURING how
well an approach works -- each of its 10 folds produces a throwaway
classifier, trained on 90% of the data, purely so it can be scored
on the held-out 10%. None of those 10 classifiers is meant to be kept.

This script is different: it trains ONE final classifier on ALL of
your labeled data (100%, no holdout), because once you're done
evaluating and ready to actually use the model to screen new,
unknown proteins, every labeled example you have is useful signal --
there's no more need to hold any of it back for scoring.

It saves two files per plastic type:
    - model_XXX.joblib        (the trained classifier itself)
    - model_XXX_metadata.json (which ESM2 model size was used, the
      embedding dimension, when it was trained, etc.)

The metadata file exists specifically to prevent a silent, hard-to-
diagnose mistake later: feeding embeddings from a DIFFERENT ESM2
model size into this classifier at prediction time. The predict
script (predict_candidates.py) reads this file and automatically
uses the matching ESM2 model, so this mismatch can't happen by
accident.
=====================================================================
"""

import json
import time
import torch
import esm
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from collections import Counter

# ---------------------------------------------------------------
# SECTION 0: CONFIGURATION
# ---------------------------------------------------------------

DATASET_FASTA = "dataset_PHA.fasta"   # <-- change per plastic type
MODEL_OUTPUT_PATH = "E:\\Urobo\\neural_net\\PET_PLA_PCL_PHA\\models\\PHA_covering_PHB\\model_PHA.joblib"
METADATA_OUTPUT_PATH = "E:\\Urobo\\neural_net\\PET_PLA_PCL_PHA\\models\\PHA_covering_PHB\\model_PHA_metadata.json"

# Which ESM2 checkpoint to use. Must match whichever size decidedly
# performed best/was most practical for this plastic type during
# earlier evaluation.
# 650 for all plastic types for consistency
ESM2_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM2_REPR_LAYER = 33


# ---------------------------------------------------------------
# SECTION 1: SETUP
# ---------------------------------------------------------------

script_start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

# esm.pretrained.<name>() is a function -- getattr() lets us call it
# by the string name in ESM2_MODEL_NAME instead of hardcoding twice
load_fn = getattr(esm.pretrained, ESM2_MODEL_NAME)
model, alphabet = load_fn()
batch_converter = alphabet.get_batch_converter()
model.eval()
model = model.to(device)


# ---------------------------------------------------------------
# SECTION 2: LOAD DATA (same FASTA parsing / labeling as before)
# ---------------------------------------------------------------

def parse_fasta(path):
    records = []
    header = None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            records.append((header, "".join(seq_lines)))
    return records


records = parse_fasta(DATASET_FASTA)
print(f"Loaded {len(records)} sequences from {DATASET_FASTA}")

labels = [0 if "label0" in header else 1 for header, seq in records]
print("Label counts:", Counter(labels))


# ---------------------------------------------------------------
# SECTION 3: EMBED EVERY SEQUENCE (identical to train_model.py)
# ---------------------------------------------------------------

embeddings = []
for i, (header, seq) in enumerate(records):
    data = [(header, seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():
        results = model(tokens, repr_layers=[ESM2_REPR_LAYER])
    emb = results["representations"][ESM2_REPR_LAYER][0, 1:len(seq) + 1].mean(0)
    embeddings.append(emb.cpu().numpy())
    if (i + 1) % 50 == 0:
        print(f"  embedded {i + 1}/{len(records)}")

X = np.array(embeddings)
y = np.array(labels)
print("Feature matrix shape:", X.shape)


# ---------------------------------------------------------------
# SECTION 4: TRAIN THE FINAL MODEL ON ALL AVAILABLE DATA
# No train/test split here -- this classifier is not being scored,
# only produced. Its expected performance was already established
# during your earlier cross-validation runs.
# ---------------------------------------------------------------

final_model = LogisticRegression(max_iter=1000)
final_model.fit(X, y)
print("Final model trained on all", len(y), "labeled proteins.")


# ---------------------------------------------------------------
# SECTION 5: SAVE THE MODEL AND ITS METADATA
# joblib is the standard way to persist scikit-learn objects --
# more efficient than Python's generic pickle for numpy-heavy
# objects like this one.
# ---------------------------------------------------------------

joblib.dump(final_model, MODEL_OUTPUT_PATH)
print(f"Saved trained model to {MODEL_OUTPUT_PATH}")

metadata = {
    "plastic_type": DATASET_FASTA.replace("dataset_", "").replace(".fasta", ""),
    "esm2_model_name": ESM2_MODEL_NAME,
    "esm2_repr_layer": ESM2_REPR_LAYER,
    "embedding_dimension": X.shape[1],
    "training_dataset_file": DATASET_FASTA,
    "n_training_proteins": len(y),
    "n_positive": int(sum(y)),
    "n_negative": int(len(y) - sum(y)),
    "trained_on": time.strftime("%Y-%m-%d %H:%M:%S"),
}
with open(METADATA_OUTPUT_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata to {METADATA_OUTPUT_PATH}")

total_elapsed = time.time() - script_start
minutes, seconds = divmod(total_elapsed, 60)
print(f"\nTotal script runtime: {int(minutes)}m {seconds:.1f}s")
