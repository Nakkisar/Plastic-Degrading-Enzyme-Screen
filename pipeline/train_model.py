"""
=====================================================================
 GENERALIZED PLASTIC-DEGRADING ENZYME CLASSIFIER -- TRAINING SCRIPT
=====================================================================
This script takes a dataset file produced by prepare_dataset.py (one
FASTA file per plastic type, containing positives / easy negatives /
hard negatives) and:

    1. Loads every protein sequence from that file.
    2. Converts each sequence into a numerical vector ("embedding")
       using a pretrained protein language model (ESM2). This is a
       neural network someone else already trained on millions of
       real proteins -- we are not training it ourselves, only using
       it as a translator from "amino acid sequence" to "list of
       numbers a simpler model can learn from."
    3. Trains a small classifier (logistic regression) on those
       vectors, using stratified k-fold cross-validation so every
       protein gets tested on exactly once, across 10 separate
       training rounds.
    4. Reports performance overall AND broken down by group
       (positive / easy negative / hard negative), plus the exact
       identity of every misclassified protein -- this is usually
       more informative than the summary numbers alone.

HOW THIS FITS INTO THE BIGGER PIPELINE:
    prepare_dataset.py  -->  dataset_XXX.fasta  -->  THIS SCRIPT
    (filters/builds        (the file this          (produces a
     the labeled data)      script reads in)        trained model +
                                                      a performance
                                                      report)

TO USE THIS FOR A DIFFERENT PLASTIC TYPE:
    Just change the DATASET_FASTA path below to point at a different
    dataset_XXX.fasta file. Nothing else needs to change.
=====================================================================
"""

import time
import torch
import esm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
from collections import Counter

# ---------------------------------------------------------------
# SECTION 0: CONFIGURATION
# This is the only line you should need to change to switch which
# plastic type you're training a model for.
# ---------------------------------------------------------------

DATASET_FASTA = "dataset_PHA.fasta"   # <-- point this at whichever dataset_XXX.fasta you want to train on

N_FOLDS = 10                          # how many cross-validation rounds to run
RANDOM_SEED = 42                      # fixes randomness so results are reproducible run-to-run


# ---------------------------------------------------------------
# SECTION 1: TIMING AND DEVICE SETUP
# Just bookkeeping -- lets you see how long the run took, and
# confirms whether your GPU is actually being used.
# ---------------------------------------------------------------

script_start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
else:
    print("Using device: cpu")


# ---------------------------------------------------------------
# SECTION 2: LOAD THE PRETRAINED PROTEIN LANGUAGE MODEL
# This downloads (first run only) and loads ESM2 -- a neural
# network already trained by Meta on tens of millions of real
# protein sequences. We use it purely as a "translator": feed it
# a protein sequence, get back a list of numbers that captures
# something meaningful about that protein's likely structure and
# function. We are NOT training this model ourselves.
# ---------------------------------------------------------------

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()             # evaluation mode -- we are only using this model, not training it
model = model.to(device)  # move the model onto the GPU (if available) for faster processing


# ---------------------------------------------------------------
# SECTION 3: READ THE DATASET FILE
# This is where the output of prepare_dataset.py enters this
# script. `records` becomes a list of (header, sequence) pairs --
# everything below this point is derived from this list.
# ---------------------------------------------------------------

def parse_fasta(path):
    """Reads a FASTA file into a list of (header, sequence) pairs."""
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

# Keep the raw header text for every protein -- this is what lets
# us print the EXACT NAME of any protein that gets misclassified
# later on, instead of just a row number.
headers = [header for header, seq in records]


# ---------------------------------------------------------------
# SECTION 4: ASSIGN LABELS
# prepare_dataset.py encodes each protein's category directly in
# its FASTA header text:
#   - contains "label0_hard"  ->  hard negative
#   - contains "label0"       ->  easy negative
#   - contains neither        ->  positive (a documented degrader)
#
# We track TWO things per protein:
#   `groups` -- the three-way category (used for the detailed
#               breakdown reporting at the end)
#   `labels` -- the simple 0/1 answer the classifier actually
#               learns to predict (both negative groups become 0)
# ---------------------------------------------------------------

def assign_group(header):
    if "label0_hard" in header:
        return "hard_negative"
    elif "label0" in header:
        return "easy_negative"
    else:
        return "positive"


groups = [assign_group(header) for header, seq in records]
labels = [0 if g in ("easy_negative", "hard_negative") else 1 for g in groups]

print("Group counts:", Counter(groups))


# ---------------------------------------------------------------
# SECTION 5: EMBED EVERY SEQUENCE
# This is the step that actually runs each protein through the
# pretrained neural network. For each protein: tokenize the raw
# letters, run them through the model, and average the resulting
# per-amino-acid vectors into a single fixed-size vector (480
# numbers) representing the whole protein -- regardless of how
# long the original sequence was.
#
# OUTPUT OF THIS SECTION: `X`, a table of shape
# (number of proteins, 480) -- one row of numbers per protein.
# This table is what the classifier in Section 6 actually learns
# from; the classifier never sees the raw amino acid letters.
# ---------------------------------------------------------------

embedding_start = time.time()

embeddings = []
for i, (header, seq) in enumerate(records):
    data = [(header, seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)
    with torch.no_grad():   # we are not training this model, so we don't need to track gradients
        results = model(tokens, repr_layers=[33])
    emb = results["representations"][33][0, 1:len(seq) + 1].mean(0)
    embeddings.append(emb.cpu().numpy())  # move the result off the GPU and into a plain numpy array
    if (i + 1) % 50 == 0:
        print(f"  embedded {i + 1}/{len(records)}")

embedding_elapsed = time.time() - embedding_start
print(f"Embedding step took {embedding_elapsed:.1f} seconds "
      f"({embedding_elapsed / len(records):.3f} sec/protein)")

X = np.array(embeddings)          # shape: (num_proteins, 480) -- the classifier's INPUT
y = np.array(labels)              # shape: (num_proteins,)     -- the classifier's TARGET (0 or 1)
groups = np.array(groups)         # kept alongside X and y for the breakdown report later
headers = np.array(headers)       # kept alongside X and y so we can name misclassified proteins later

print("\nFeature matrix shape:", X.shape)


# ---------------------------------------------------------------
# SECTION 6: TRAIN AND EVALUATE WITH STRATIFIED K-FOLD CROSS-VALIDATION
# Rather than a single train/test split (which risks an unlucky or
# lucky split skewing the result), the dataset is divided into
# N_FOLDS chunks. Across N_FOLDS rounds, each chunk gets a turn
# being the "held out" test set while the classifier trains fresh
# on the rest. Every protein ends up tested on EXACTLY ONCE, but
# each individual round still trains on the large majority of the
# data.
#
# "Stratified" means each fold is built to contain the same
# proportion of positive / easy-negative / hard-negative proteins
# as the full dataset -- this prevents a fold from accidentally
# ending up with almost no hard negatives (or no positives) purely
# by chance, which would make that round's difficulty inconsistent
# with the others.
#
# INPUT USED: `X`, `y`, `groups` from Section 5.
# OUTPUT OF THIS SECTION: `oof_preds` ("out-of-fold predictions")
# -- one prediction per protein, gathered from whichever round
# happened to test it. This feeds directly into Section 7 below,
# where we report performance broken down by group and list every
# misclassified protein by name.
# ---------------------------------------------------------------

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

oof_preds = np.zeros(len(y), dtype=int)   # will be filled in as each fold runs

fold_mcc, fold_f1, fold_precision, fold_recall = [], [], [], []

# skf.split(X, groups) hands back N_FOLDS pairs of (train_idx, test_idx) --
# lists of ROW NUMBERS telling us which proteins go into training vs
# testing for that particular round.
for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, groups), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # A brand new, untrained classifier every round -- nothing carries
    # over between folds, so each round is a genuinely independent test.
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    # Record each tested protein's prediction in its permanent slot.
    oof_preds[test_idx] = preds

    mcc = matthews_corrcoef(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)

    fold_mcc.append(mcc)
    fold_f1.append(f1)
    fold_precision.append(precision)
    fold_recall.append(recall)

    print(f"Fold {fold_num:2d}:  MCC={mcc:.2f}  F1={f1:.2f}  Precision={precision:.2f}  Recall={recall:.2f}")

print("\n--- Mean across all folds (overall) ---")
print(f"MCC:       {np.mean(fold_mcc):.3f}")
print(f"F1:        {np.mean(fold_f1):.3f}")
print(f"Precision: {np.mean(fold_precision):.3f}")
print(f"Recall:    {np.mean(fold_recall):.3f}")


# ---------------------------------------------------------------
# SECTION 7: BREAKDOWN BY GROUP + NAME EVERY MISCLASSIFIED PROTEIN
# The overall average above can look good even if the model is
# only doing well on the "easy" part of the task. This section
# checks performance SEPARATELY for positives / easy negatives /
# hard negatives, and prints the exact identity of every protein
# the model got wrong -- this is usually the most useful output
# of the whole script for deciding what to investigate next.
#
# INPUT USED: `oof_preds` from Section 6, `groups`/`headers` from
# Section 4/5.
# ---------------------------------------------------------------

print("\n--- Breakdown by group (using each protein's out-of-fold prediction) ---")

for group_name in ["positive", "easy_negative", "hard_negative"]:
    mask = groups == group_name
    n = mask.sum()

    if n == 0:
        print(f"{group_name:15s} (n=   0):  no samples in this group -- skipping")
        continue

    pred_vals = oof_preds[mask]

    if group_name == "positive":
        correctly_identified = (pred_vals == 1).sum()
        print(f"{group_name:15s} (n={n:4d}):  correctly predicted positive = "
              f"{correctly_identified}/{n}  ({correctly_identified / n:.1%})")
    else:
        correctly_identified = (pred_vals == 0).sum()
        false_positives = (pred_vals == 1).sum()
        print(f"{group_name:15s} (n={n:4d}):  correctly predicted negative = "
              f"{correctly_identified}/{n}  ({correctly_identified / n:.1%})   "
              f"false positives = {false_positives}")

print("\n--- Misclassified entries ---")

# Proteins that ARE documented degraders, but the model predicted "not a degrader"
mask = (groups == "positive") & (oof_preds == 0)
misses = headers[mask]
print(f"\nMissed degraders (false negatives): {len(misses)}")
for h in misses:
    print(f"  {h}")

# Easy negatives the model incorrectly flagged as degraders
mask = (groups == "easy_negative") & (oof_preds == 1)
misses = headers[mask]
print(f"\nEasy negatives wrongly flagged as degrading (false positives): {len(misses)}")
for h in misses:
    print(f"  {h}")

# Hard negatives the model incorrectly flagged as degraders --
# usually the most interesting group to look at closely, since
# these are the cases closest to the real decision boundary.
mask = (groups == "hard_negative") & (oof_preds == 1)
misses = headers[mask]
print(f"\nHard negatives wrongly flagged as degrading (false positives): {len(misses)}")
for h in misses:
    print(f"  {h}")


# ---------------------------------------------------------------
# SECTION 8: TOTAL RUNTIME
# ---------------------------------------------------------------

total_elapsed = time.time() - script_start
minutes, seconds = divmod(total_elapsed, 60)
print(f"\nTotal script runtime: {int(minutes)}m {seconds:.1f}s")
