# Plastic-Degrading-Enzyme-Screen
Neural Network pipeline to identify likely plastic degrading enzymes from FASTA files containing protein sequences.

This is a sequence-embedding classifier pipeline for identifying candidate plastic-degrading enzymes (PET, PLA, PCL, and PHA/PHB) directly from protein sequence, at whole-proteome scale.

This tool takes any FASTA file of protein sequences (a genome, a metagenomic sample, a curated database export — anything) and, for each of four bioplastic types, produces a ranked list of the proteins most likely to be a degrading enzyme for that plastic. It's built to be a fast triage step: narrowing a large, unannotated candidate pool down to a short, prioritized list worth investigating further, ahead of slower structural or wet-lab work.

It does not replace structural modeling, docking, or experimental validation — it's a complementary first pass, not a replacement for either.

Background

The methodology follows a University of Padova master's thesis (Minto, 2023–2024), which developed a semi-supervised classifier for identifying PET-degrading enzymes from sequence embeddings. This project follows the same general structure, with database-derived positive/negative labeling, sequence embedding via a pretrained protein language model, cross-validated classification (extended to three additional plastic types), with a larger current dataset snapshot, and a deliberately harder two-tier negative set (see Methodology below).

Structural information (e.g. AlphaFold-derived embeddings) was deliberately not included. The original thesis tested a combined sequence-and-structure approach for PET and found it did not meaningfully improve classification performance over sequence alone. This project followed that precedent to keep the pipeline simpler and faster.

Validation: an independent real-world case study

Beyond database cross-validation, this project's models were tested against a genuinely independent case: the complete, unannotated proteome (10,035 proteins) of _Coniochaeta pulveracea_ strain CAB683, screened using an entirely separate, structure-guided discovery pipeline (TM-Vec/GASS/docking) in an unrelated thesis. All five of that thesis's shortlisted candidates ranked within the top 1–17% of the full genome on at least one plastic-type model here, despite the two approaches sharing no training data or prior knowledge of each other's results. One candidate (RKU40824.1) received the single strongest computational signal of any candidate tested — ranked 4th of 10,035 proteins for PCL — despite an inconclusive wet-lab result, illustrating the kind of high-confidence, prioritizable candidate this tool is meant to surface.

Methodology
1. Positives — pulled from PlasticDB, filtered per plastic type. A protein documented as degrading multiple plastics counts as a positive for every plastic type it's tagged with.
2. Easy negatives — random reviewed UniProt/SwissProt proteins, excluding enzyme classes (by EC number) broadly associated with plastic degradation.
3. Hard negatives — real cutinases/esterases/lipases (the enzyme family most plastic-degraders belong to) not documented as degrading the target plastic, specifically to force the classifier to learn a finer distinction than "is this a hydrolase at all."
4. Embedding — ESM2 (Meta AI), a pretrained protein language model, used purely as a fixed sequence-to-vector translator; no fine-tuning.
5. Classification — logistic regression, evaluated with stratified 10-fold cross-validation, validated further with full per-protein misclassification review rather than aggregate metrics alone.
6. Deployment — each saved model is retrained on 100% of its labeled dataset (cross-validation folds are for measuring performance, not for producing the model actually used to screen new proteins).

Known limitations
Training data is predominantly bacterial in origin; performance on distant taxa (e.g. fungi) is promising in the CAB683 case study above but not exhaustively validated.
Recurring blind spot for broad-spectrum, multi-substrate generalist enzymes and near-identical sequence paralogs. These proteins most likely to need a second look rather than being taken at face value in either direction as degrader or non-degrader.
This is a sequence-only classifier. A structurally-informed model has not been tested against these specific targets and might behave differently.
Predictions are a prioritization aid, not a confirmation of activity. Treat a high score as "worth investigating," not "confirmed."
Data sources and licensing
ESM2 (Meta AI, via fair-esm) — MIT license, code and weights, explicitly permitted for commercial use. Note: this is distinct from newer EvolutionaryScale models (ESM3, ESM C), which carry different, more restrictive non-commercial licensing.
PlasticDB (Gambarini et al., 2022) — published CC BY 4.0; attribution required.
UniProt/SwissProt — CC BY 4.0; explicitly available for commercial and non-commercial use; attribution required.
Third-party libraries (PyTorch, scikit-learn, joblib, Pillow) — standard permissive open-source licenses.
