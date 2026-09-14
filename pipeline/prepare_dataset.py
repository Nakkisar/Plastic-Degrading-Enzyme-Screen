"""
=====================================================================
 GENERALIZED DATASET PREPARATION PIPELINE
=====================================================================
This script builds a complete labeled dataset (positives + easy
negatives + hard negatives) for ONE target plastic type at a time.
It is a generalized version of the PET-specific pipeline built
earlier -- the only thing that changes between plastic types is the
`PLASTIC_TYPE` string at the top.

WHAT THIS SCRIPT PRODUCES:
    A single FASTA file per plastic type (e.g. "dataset_PLA.fasta")
    containing:
      - POSITIVES: enzymes from PlasticDB documented as degrading
        this plastic type (a protein documented as degrading
        MULTIPLE plastics, e.g. "PCL_PET_PHB", counts as a positive
        for EACH of those plastic types individually).
      - EASY NEGATIVES: random SwissProt proteins, filtered to
        exclude enzyme classes (by EC number) broadly associated
        with plastic/polyester degradation.
      - HARD NEGATIVES: real cutinases / esterases / lipases (the
        enzyme family most plastic-degraders belong to) that are
        NOT documented as degrading this specific plastic type.

    This output FASTA is the INPUT to the next stage of the
    pipeline: train_model.py, which reads this file, converts each
    sequence into a numerical vector, and trains a classifier on it.
=====================================================================
"""

import csv
import random

# ---------------------------------------------------------------
# SECTION 0: CONFIGURATION
# Change PLASTIC_TYPE to build a dataset for a different plastic.
# Everything below this point runs automatically based on that
# one setting -- nothing else needs to be edited per plastic type.
# ---------------------------------------------------------------

PLASTIC_TYPE = "PHA"

PLASTICDB_FASTA = "E:\\Urobo\\PlasticDB_09_03_2026\\PlasticDB.fasta"
SWISSPROT_TSV = "E:\\Urobo\\neural_net\\uniprot_negatives\\uniprotkb_reviewed_true_AND_existence_1_2026_08_18.tsv"
HARD_NEGATIVE_CANDIDATES_TSV = "E:\\Urobo\\neural_net\\uniprot_negatives\\uniprotkb_cutinase_OR_carboxylesterase_2026_08_19.tsv"

OUTPUT_FASTA = f"E:\\Urobo\\neural_net\\PET_PLA_PCL_PHA\\datasets_plastic_type\\dataset_{PLASTIC_TYPE}.fasta"

EASY_NEGATIVE_RATIO = 2    # how many "easy negatives" to sample per positive
HARD_NEGATIVE_RATIO = 1    # how many "hard negatives" to sample per positive
                            # (kept as a RATIO, not a fixed count, so the class
                            # balance stays consistent across plastic types with
                            # very different numbers of known positives -- e.g.
                            # PLA has far fewer positives than PET, so a FIXED
                            # hard-negative count would swamp PLA's dataset with
                            # a much harsher imbalance than PET was trained under)
RANDOM_SEED = 42

# EC number prefixes (first three digits, e.g. "3.1.1") broadly
# associated with plastic/polyester-degrading enzyme activity.
# Any candidate "easy negative" whose EC number falls in one of
# these families is excluded, so the easy-negative pool doesn't
# accidentally contain a protein that's secretly a plastic degrader.
# NOTE: this list was built from the EC numbers documented in the
# source thesis for PET-degrading enzymes specifically. It is
# applied here to ALL plastic types as a broad, conservative filter,
# since most bioplastic-degrading enzymes (PLA, PCL, PHA/PHB
# depolymerases included) fall in the same general esterase/
# hydrolase superfamily. This is a reasonable approximation, not a
# guarantee of completeness for plastic types other than PET.
EXCLUDED_EC_PREFIXES_FOR_EASY_NEGATIVES = {"2.5.1", "3.1.1", "3.2.1"}

# Literal EC numbers known to correspond DIRECTLY to a specific
# plastic-degrading activity. Any hard-negative candidate carrying
# one of these EC numbers is excluded outright, regardless of
# whether PlasticDB happens to have a matching entry for it yet,
# because the EC number itself confirms the activity.
# NOTE: this is currently only populated with confidently-known
# entries for PET (from the thesis + our own data exploration).
# For PLA/PCL/PHA/PHB we do not have an equally authoritative list,
# so this acts as a best-effort safety net for those plastic types --
# the SEQUENCE COLLISION CHECK (Section 3 below) remains the primary
# safeguard against mislabeling for every plastic type, including PET.
KNOWN_DIRECT_DEGRADATION_EC_NUMBERS = {
    "PET": {"3.1.1.101", "3.1.1.102"},   # PETase, MHETase
}


# ---------------------------------------------------------------
# SECTION 1: SHARED HELPER FUNCTIONS
# These are small, reusable building blocks used throughout the
# rest of the script. Read once, used many times below.
# ---------------------------------------------------------------

def parse_fasta(path):
    """
    Reads a FASTA file and returns a list of (header, sequence) pairs.
    A FASTA file looks like:
        >header text
        SEQUENCELETTERS
        >another header
        MORESEQUENCE
    This function turns that text format into a Python list you can
    loop over normally.
    """
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


def ec_prefix3(ec):
    """Returns the first three dot-separated numbers of an EC code, e.g. '3.1.1.74' -> '3.1.1'."""
    parts = ec.strip().split(".")
    return ".".join(parts[:3]) if len(parts) >= 3 else None


# ---------------------------------------------------------------
# SECTION 2: BUILD THE POSITIVE SET
# Reads the full PlasticDB file and keeps only entries tagged with
# our target plastic type. A protein tagged with multiple plastic
# types (e.g. "PCL_PET_PHB") is kept as a positive for ALL of them --
# so running this script once per plastic type will legitimately
# reuse some of the same proteins across different output files.
# That is intentional and reflects real biology: many of these
# enzymes are broad-spectrum polyester hydrolases.
#
# OUTPUT OF THIS SECTION: `positive_records`, a list of
# (header, sequence) pairs -- this feeds into Section 4 (dedup +
# collision-checking) below.
# ---------------------------------------------------------------

all_plasticdb_records = parse_fasta(PLASTICDB_FASTA)

positive_records = []
for header, seq in all_plasticdb_records:
    parts = header.split("|")
    if len(parts) >= 4:
        plastic_tokens = parts[3].split("_")
        if PLASTIC_TYPE in plastic_tokens:
            positive_records.append((header, seq))

# Remove exact-duplicate sequences (the same protein catalogued
# more than once under different PlasticDB IDs).
seen_sequences = set()
deduped_positive_records = []
for header, seq in positive_records:
    if seq not in seen_sequences:
        seen_sequences.add(seq)
        deduped_positive_records.append((header, seq))

positive_records = deduped_positive_records
positive_sequences = set(seq for header, seq in positive_records)

print(f"[{PLASTIC_TYPE}] Positive set: {len(positive_records)} unique sequences")


# ---------------------------------------------------------------
# SECTION 3: BUILD THE EASY-NEGATIVE SET
# Randomly samples reviewed SwissProt proteins, excluding any whose
# EC number falls in a plastic-degradation-associated family (see
# EXCLUDED_EC_PREFIXES_FOR_EASY_NEGATIVES above), and excluding any
# that happen to have the exact same sequence as one of our
# positives (a safety net -- catches cases where the "random"
# SwissProt protein IS actually one of our target enzymes).
#
# INPUT USED: `positive_sequences`, produced in Section 2 above.
# OUTPUT OF THIS SECTION: `easy_negative_records` -- feeds into
# Section 5 (writing the final combined file).
# ---------------------------------------------------------------

def has_excluded_ec(ec_field):
    if not ec_field.strip():
        return False  # no EC number at all -> not an enzyme -> safe to keep
    for ec in ec_field.split(";"):
        if ec_prefix3(ec) in EXCLUDED_EC_PREFIXES_FOR_EASY_NEGATIVES:
            return True
    return False


with open(SWISSPROT_TSV) as f:
    reader = csv.DictReader(f, delimiter="\t")
    swissprot_rows = list(reader)

easy_negative_candidates = [
    r for r in swissprot_rows
    if not has_excluded_ec(r["EC number"]) and r["Sequence"] not in positive_sequences
]

target_n_easy = len(positive_records) * EASY_NEGATIVE_RATIO
random.seed(RANDOM_SEED)
sampled_easy_negatives = random.sample(
    easy_negative_candidates,
    min(target_n_easy, len(easy_negative_candidates))
)

easy_negative_records = []
for r in sampled_easy_negatives:
    header = f"{r['Entry']}|{r['Entry Name']}|{r['Organism']}|EC={r['EC number'] or 'none'}|label0"
    easy_negative_records.append((header, r["Sequence"]))

print(f"[{PLASTIC_TYPE}] Easy negative set: {len(easy_negative_records)} sequences")


# ---------------------------------------------------------------
# SECTION 4: BUILD THE HARD-NEGATIVE SET
# Pulls from a pool of real cutinases / esterases / lipases (the
# enzyme family most plastic-degraders belong to). These are
# "hard" negatives because they are functionally much closer to
# true plastic-degraders than the easy negatives above -- forcing
# the eventual classifier to learn a finer distinction than just
# "is this a hydrolase at all".
#
# Three exclusion checks are applied, each guarding against a
# different way a "hard negative" candidate could actually be a
# mislabeled positive:
#   1. Wrong EC family entirely (keeps only the true 3.1.1 family)
#   2. A literal, confirmed plastic-degradation EC number (see
#      KNOWN_DIRECT_DEGRADATION_EC_NUMBERS above)
#   3. Exact sequence match with one of our confirmed positives
#      (the strongest, most reliable check -- applies regardless
#      of whether we know the "correct" EC number in advance)
#
# INPUT USED: `positive_sequences`, from Section 2.
# OUTPUT OF THIS SECTION: `hard_negative_records` -- feeds into
# Section 5 (writing the final combined file), same as easy negatives.
# ---------------------------------------------------------------

def is_true_311_family(ec_field):
    for ec in ec_field.split(";"):
        if ec_prefix3(ec) == "3.1.1":
            return True
    return False


def has_known_direct_degradation_ec(ec_field, plastic_type):
    known_ecs = KNOWN_DIRECT_DEGRADATION_EC_NUMBERS.get(plastic_type, set())
    for ec in ec_field.split(";"):
        if ec.strip() in known_ecs:
            return True
    return False


with open(HARD_NEGATIVE_CANDIDATES_TSV) as f:
    reader = csv.DictReader(f, delimiter="\t")
    hard_negative_rows = list(reader)

hard_negative_candidates = []
for r in hard_negative_rows:
    ec_field = r["EC number"]
    seq = r["Sequence"]
    if not is_true_311_family(ec_field):
        continue
    if has_known_direct_degradation_ec(ec_field, PLASTIC_TYPE):
        continue
    if seq in positive_sequences:
        continue
    hard_negative_candidates.append(r)

target_n_hard = len(positive_records) * HARD_NEGATIVE_RATIO
sampled_hard_negatives = random.sample(
    hard_negative_candidates,
    min(target_n_hard, len(hard_negative_candidates))
)

hard_negative_records = []
for r in sampled_hard_negatives:
    header = f"{r['Entry']}|{r['Entry Name']}|{r['Organism']}|EC={r['EC number']}|label0_hard"
    hard_negative_records.append((header, r["Sequence"]))

print(f"[{PLASTIC_TYPE}] Hard negative set: {len(hard_negative_records)} sequences "
      f"(from {len(hard_negative_candidates)} valid candidates)")


# ---------------------------------------------------------------
# SECTION 5: COMBINE AND WRITE THE FINAL DATASET FILE
# This is the file train_model.py (the next stage of the pipeline)
# will read. Positives are written first, then easy negatives,
# then hard negatives -- order doesn't matter for training (the
# training script shuffles anyway), but keeping it consistent
# makes the file easier to skim by eye.
# ---------------------------------------------------------------

all_records = positive_records + easy_negative_records + hard_negative_records

with open(OUTPUT_FASTA, "w") as f:
    for header, seq in all_records:
        f.write(f">{header}\n{seq}\n")

print(f"[{PLASTIC_TYPE}] Wrote {len(all_records)} total sequences to {OUTPUT_FASTA}")
print(f"[{PLASTIC_TYPE}]   -> {len(positive_records)} positive / "
      f"{len(easy_negative_records)} easy negative / {len(hard_negative_records)} hard negative")
