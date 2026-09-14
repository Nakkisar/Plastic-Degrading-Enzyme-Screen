"""
=====================================================================
 GUI FOR SCREENING NEW / UNKNOWN PROTEINS AGAINST A SAVED MODEL
 (with GPU/CPU selection and reusable embedding caching)
=====================================================================
This is a desktop window (built with tkinter, bundled with Python --
no extra installation needed) that lets you:

    1. Pick a saved model file (a .joblib file from save_final_model.py)
    2. Pick a FASTA file of candidate/unknown proteins to screen
    3. Choose whether to force CPU-only mode (useful if sharing this
       with someone who has no GPU)
    4. Choose whether to reuse a previously saved set of embeddings
       instead of re-embedding from scratch
    5. Click "Run Prediction" and watch progress in a log window

WHY EMBEDDING CACHING MATTERS HERE SPECIFICALLY:
    If you use the SAME ESM2 model (e.g. the 650M version) for all
    four plastic-type classifiers, then embedding a given batch of
    candidate proteins produces the EXACT SAME numbers regardless of
    which plastic-type model you're about to apply afterward -- only
    the final classification step differs between plastic types.
    Embedding is by far the slowest part of this whole process, so
    saving it once and reusing it for all four plastic types turns
    four slow runs into one slow run plus three fast ones.

    The cache file records exactly which ESM2 model produced it, and
    a fingerprint (hash) of the exact FASTA file it was built from --
    so if you ever try to reuse a cache with a mismatched model or a
    different/changed FASTA file, the script will refuse with a clear
    error rather than silently producing meaningless predictions.

WHY A BACKGROUND THREAD IS USED:
    Embedding can take anywhere from seconds to minutes. If that work
    ran directly on a button click, the window would freeze for the
    whole duration. So the actual work happens on a separate
    background thread, while the main thread stays free to keep the
    window responsive. Tkinter widgets are not safe to touch directly
    from a background thread, so progress messages are placed on a
    thread-safe queue.Queue, and the main thread copies them into the
    log box roughly 10 times per second.
=====================================================================
"""

import json
import time
import queue
import threading
import csv
import os
import hashlib

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import torch
import esm
import numpy as np
import joblib
from PIL import Image, ImageTk


# ---------------------------------------------------------------
# SECTION 1: SMALL HELPER FUNCTIONS
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


def read_metadata(model_path):
    """Loads the metadata JSON that save_final_model.py saves alongside a model."""
    metadata_path = model_path.replace(".joblib", "_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Could not find a matching metadata file at:\n{metadata_path}\n"
            f"Make sure it sits alongside the .joblib file, with the same name."
        )
    with open(metadata_path) as f:
        return json.load(f)


def compute_file_hash(path):
    """Returns a short fingerprint of a file's exact contents, used to detect
    if a cached embeddings file no longer matches the FASTA it was built from."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def default_cache_path(fasta_path, esm2_model_name):
    """Builds a predictable, descriptive default filename for a saved
    embeddings cache, placed next to the source FASTA file."""
    base, _ = os.path.splitext(fasta_path)
    return f"{base}__{esm2_model_name}_embeddings.joblib"


# --- Logo configuration ---
# Point LOGO_PATH at an image file (PNG, JPG, etc.) to display it in the
# top-right corner of the window. If the file isn't found, the window
# simply opens without a logo -- so this is safe to leave in the script
# even when sharing a copy that doesn't include the image file.
LOGO_PATH = "E:\\Urobo\\placeholder.png"   # <-- change this to your logo's filename/path
LOGO_MAX_HEIGHT = 50              # pixels -- the logo is scaled to this height,
                                   # preserving its original aspect ratio


# ---------------------------------------------------------------
# SECTION 2: EMBEDDING (fresh) AND CACHING (save / load)
# ---------------------------------------------------------------

def embed_sequences(records, esm2_model_name, esm2_repr_layer, device, log):
    """Runs every (header, sequence) pair through ESM2 and returns a
    numpy array of shape (num_proteins, embedding_dimension)."""
    load_fn = getattr(esm.pretrained, esm2_model_name)
    esm_model, alphabet = load_fn()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()
    esm_model = esm_model.to(device)

    embeddings = []
    for i, (header, seq) in enumerate(records):
        data = [(header, seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = esm_model(tokens, repr_layers=[esm2_repr_layer])
        emb = results["representations"][esm2_repr_layer][0, 1:len(seq) + 1].mean(0)
        embeddings.append(emb.cpu().numpy())
        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            log(f"  embedded {i + 1}/{len(records)}")
    return np.array(embeddings)


def save_embedding_cache(cache_path, headers, seq_lengths, X, esm2_model_name,
                          esm2_repr_layer, source_fasta_path, log):
    cache = {
        "headers": headers,
        "sequence_lengths": seq_lengths,
        "embeddings": X,
        "esm2_model_name": esm2_model_name,
        "esm2_repr_layer": esm2_repr_layer,
        "embedding_dimension": X.shape[1],
        "source_fasta_path": source_fasta_path,
        "source_fasta_hash": compute_file_hash(source_fasta_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(cache, cache_path)
    log(f"Saved reusable embeddings cache to:\n{cache_path}\n"
        f"(valid for any plastic-type model that also uses {esm2_model_name})")


def load_embedding_cache(cache_path, source_fasta_path, expected_esm2_model_name,
                          expected_embedding_dim, log):
    cache = joblib.load(cache_path)

    # Guard 1: was this cache built from the exact same FASTA file's contents?
    current_hash = compute_file_hash(source_fasta_path)
    if cache.get("source_fasta_hash") != current_hash:
        raise ValueError(
            "The selected embeddings cache does not match the selected FASTA file "
            "(the file's contents differ from what the cache was built from -- "
            "either a different file, or this one has changed since). "
            "Uncheck 'Use cached embeddings' to re-embed, or pick the correct cache file."
        )

    # Guard 2: was this cache built with the SAME ESM2 model this classifier expects?
    if cache.get("esm2_model_name") != expected_esm2_model_name:
        raise ValueError(
            f"This cache was built with '{cache.get('esm2_model_name')}', but the "
            f"selected model expects '{expected_esm2_model_name}'. These embeddings "
            f"are not compatible with each other."
        )
    if cache.get("embedding_dimension") != expected_embedding_dim:
        raise ValueError("Embedding dimension mismatch between the cache and the model.")

    log(f"Loaded cached embeddings from:\n{cache_path}\n"
        f"({len(cache['headers'])} proteins) -- skipping re-embedding.")
    return cache["headers"], cache["sequence_lengths"], cache["embeddings"]


# ---------------------------------------------------------------
# SECTION 3: THE CORE PREDICTION WORKFLOW
# ---------------------------------------------------------------

def run_prediction(model_path, candidate_fasta_path, output_csv_path, log,
                    device_choice="Auto-detect (use GPU if available)",
                    use_cached_embeddings=False, cached_embeddings_path=None,
                    save_embeddings_after=True):

    script_start = time.time()

    metadata = read_metadata(model_path)
    log(f"Loaded model metadata:")
    log(f"  Plastic type: {metadata['plastic_type']}")
    log(f"  ESM2 model:   {metadata['esm2_model_name']}")
    log(f"  Trained on:   {metadata['n_training_proteins']} proteins "
        f"({metadata['n_positive']} positive / {metadata['n_negative']} negative)")

    classifier = joblib.load(model_path)
    log(f"Loaded classifier from {model_path}")

    esm2_model_name = metadata["esm2_model_name"]
    esm2_repr_layer = metadata["esm2_repr_layer"]
    expected_embedding_dim = metadata["embedding_dimension"]

    # --- Device selection: honor an explicit "Force CPU" choice from the GUI ---
    if device_choice == "Force CPU":
        device = torch.device("cpu")
        log("Using device: cpu (forced by user selection)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            log(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
        else:
            log("Using device: cpu (no GPU detected)")

    candidates = parse_fasta(candidate_fasta_path)
    log(f"Loaded {len(candidates)} candidate proteins from {candidate_fasta_path}")

    headers = [h for h, s in candidates]
    seq_lengths = [len(s) for h, s in candidates]

    # --- Either load a cached embedding, or compute one fresh ---
    if use_cached_embeddings:
        if not cached_embeddings_path or not os.path.exists(cached_embeddings_path):
            raise FileNotFoundError(
                "Please select a valid embeddings cache file, or uncheck "
                "'Use cached embeddings' to embed from scratch."
            )
        headers, seq_lengths, X_candidates = load_embedding_cache(
            cached_embeddings_path, candidate_fasta_path,
            esm2_model_name, expected_embedding_dim, log
        )
    else:
        embedding_start = time.time()
        X_candidates = embed_sequences(candidates, esm2_model_name, esm2_repr_layer, device, log)
        log(f"Embedding step took {time.time() - embedding_start:.1f} seconds")

        if save_embeddings_after:
            cache_path = default_cache_path(candidate_fasta_path, esm2_model_name)
            save_embedding_cache(
                cache_path, headers, seq_lengths, X_candidates,
                esm2_model_name, esm2_repr_layer, candidate_fasta_path, log
            )

    if X_candidates.shape[1] != expected_embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: got {X_candidates.shape[1]}, "
            f"expected {expected_embedding_dim}."
        )

    # --- Predict and rank ---
    probabilities = classifier.predict_proba(X_candidates)[:, 1]

    result_rows = []
    for header, seq_len, prob in zip(headers, seq_lengths, probabilities):
        result_rows.append({
            "header": header,
            "sequence_length": seq_len,
            "predicted_probability": round(float(prob), 4),
            "predicted_label": "degrader" if prob >= 0.5 else "non-degrader",
        })
    result_rows.sort(key=lambda r: r["predicted_probability"], reverse=True)

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["header", "sequence_length", "predicted_probability", "predicted_label"])
        writer.writeheader()
        writer.writerows(result_rows)

    log(f"\nWrote ranked predictions for {len(result_rows)} candidates to:\n{output_csv_path}")

    n_flagged = sum(1 for r in result_rows if r["predicted_label"] == "degrader")
    log(f"{n_flagged}/{len(result_rows)} candidates predicted as likely {metadata['plastic_type']}-degraders")

    log("\nTop 10 candidates by predicted probability:")
    for r in result_rows[:10]:
        log(f"  {r['predicted_probability']:.3f}  {r['header'][:70]}")

    total_elapsed = time.time() - script_start
    minutes, seconds = divmod(total_elapsed, 60)
    log(f"\nTotal runtime: {int(minutes)}m {seconds:.1f}s")

    return output_csv_path, n_flagged, len(result_rows)


# ---------------------------------------------------------------
# SECTION 4: THE GUI ITSELF
# ---------------------------------------------------------------

class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plastic-Degrader Enzyme Screener")
        self.root.geometry("740x680")

        self.log_queue = queue.Queue()

        self.model_path = tk.StringVar()
        self.fasta_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.device_choice = tk.StringVar(value="Auto-detect (use GPU if available)")
        self.save_cache_var = tk.BooleanVar(value=True)
        self.use_cache_var = tk.BooleanVar(value=False)
        self.cache_load_path = tk.StringVar()

        # Keeping this on self.logo_image is required, not optional --
        # tkinter does not hold its own reference to image objects, so
        # without storing it somewhere Python's garbage collector would
        # free the image shortly after it's drawn, and it would vanish
        # from the window (a very common, confusing tkinter gotcha).
        self.logo_image = self._load_logo()

        self._build_widgets()
        self.root.after(100, self._poll_log_queue)

    def _load_logo(self):
        """Loads and proportionally resizes the logo image, if the file
        exists. Returns None (and the window simply shows no logo) if
        LOGO_PATH doesn't point at a real file -- so this never crashes
        the app just because the image is missing."""
        if not os.path.exists(LOGO_PATH):
            return None
        try:
            image = Image.open(LOGO_PATH)
            width, height = image.size
            scale = LOGO_MAX_HEIGHT / height
            resized = image.resize((int(width * scale), LOGO_MAX_HEIGHT))
            return ImageTk.PhotoImage(resized)
        except Exception:
            return None

    def _build_widgets(self):
        padding = {"padx": 10, "pady": 6}

        # --- Header row: app title on the left, logo (if available) on the right ---
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=10, pady=(10, 0))
        ttk.Label(
            header_frame, text="Plastic-Degrader Enzyme Screener",
            font=("TkDefaultFont", 13, "bold")
        ).pack(side="left")
        if self.logo_image is not None:
            ttk.Label(header_frame, image=self.logo_image).pack(side="right")

        # --- Model file picker ---
        frame1 = ttk.Frame(self.root)
        frame1.pack(fill="x", **padding)
        ttk.Label(frame1, text="Saved model (.joblib):", width=24).pack(side="left")
        ttk.Entry(frame1, textvariable=self.model_path).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(frame1, text="Browse...", command=self._browse_model).pack(side="left")

        # --- Candidate FASTA file picker ---
        frame2 = ttk.Frame(self.root)
        frame2.pack(fill="x", **padding)
        ttk.Label(frame2, text="Candidate FASTA file:", width=24).pack(side="left")
        ttk.Entry(frame2, textvariable=self.fasta_path).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(frame2, text="Browse...", command=self._browse_fasta).pack(side="left")

        # --- Output CSV picker ---
        frame3 = ttk.Frame(self.root)
        frame3.pack(fill="x", **padding)
        ttk.Label(frame3, text="Output CSV:", width=24).pack(side="left")
        ttk.Entry(frame3, textvariable=self.output_path).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Button(frame3, text="Choose...", command=self._browse_output).pack(side="left")

        # --- Device selector ---
        frame_device = ttk.Frame(self.root)
        frame_device.pack(fill="x", **padding)
        ttk.Label(frame_device, text="Compute device:", width=24).pack(side="left")
        device_dropdown = ttk.Combobox(
            frame_device, textvariable=self.device_choice,
            values=["Auto-detect (use GPU if available)", "Force CPU"],
            state="readonly", width=35,
        )
        device_dropdown.pack(side="left")

        # --- Embedding cache controls ---
        cache_box = ttk.LabelFrame(self.root, text="Embedding reuse (saves time across multiple plastic-type models)")
        cache_box.pack(fill="x", padx=10, pady=8)

        save_row = ttk.Frame(cache_box)
        save_row.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Checkbutton(
            save_row, text="Save embeddings after this run, for reuse with other models",
            variable=self.save_cache_var
        ).pack(side="left")

        load_row = ttk.Frame(cache_box)
        load_row.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Checkbutton(
            load_row, text="Use previously saved embeddings (skip re-embedding)",
            variable=self.use_cache_var, command=self._toggle_cache_controls
        ).pack(side="left")

        load_row2 = ttk.Frame(cache_box)
        load_row2.pack(fill="x", padx=8, pady=(0, 8))
        self.cache_entry = ttk.Entry(load_row2, textvariable=self.cache_load_path, state="disabled")
        self.cache_entry.pack(side="left", fill="x", expand=True, padx=(24, 5))
        self.cache_browse_button = ttk.Button(load_row2, text="Browse...", command=self._browse_cache, state="disabled")
        self.cache_browse_button.pack(side="left")

        # --- Run button + progress bar ---
        frame4 = ttk.Frame(self.root)
        frame4.pack(fill="x", **padding)
        self.run_button = ttk.Button(frame4, text="Run Prediction", command=self._on_run_clicked)
        self.run_button.pack(side="left")
        self.progress_bar = ttk.Progressbar(frame4, mode="indeterminate")
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=10)

        # --- Log output area ---
        ttk.Label(self.root, text="Progress log:").pack(anchor="w", padx=10)
        self.log_box = scrolledtext.ScrolledText(self.root, height=20, state="disabled", wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # --- Cache checkbox enable/disable logic ---

    def _toggle_cache_controls(self):
        if self.use_cache_var.get():
            self.cache_entry.config(state="normal")
            self.cache_browse_button.config(state="normal")
        else:
            self.cache_entry.config(state="disabled")
            self.cache_browse_button.config(state="disabled")

    # --- File dialog callbacks ---

    def _browse_model(self):
        path = filedialog.askopenfilename(title="Select a saved model", filetypes=[("Joblib model files", "*.joblib")])
        if path:
            self.model_path.set(path)
            self._maybe_autofill_output()

    def _browse_fasta(self):
        path = filedialog.askopenfilename(title="Select a candidate FASTA file", filetypes=[("FASTA files", "*.fasta *.fa"), ("All files", "*.*")])
        if path:
            self.fasta_path.set(path)
            self._maybe_autofill_output()

    def _browse_output(self):
        path = filedialog.asksaveasfilename(title="Save results as...", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            self.output_path.set(path)

    def _browse_cache(self):
        path = filedialog.askopenfilename(title="Select a saved embeddings cache", filetypes=[("Joblib cache files", "*.joblib")])
        if path:
            self.cache_load_path.set(path)

    def _maybe_autofill_output(self):
        if self.model_path.get() and self.fasta_path.get() and not self.output_path.get():
            model_name = os.path.splitext(os.path.basename(self.model_path.get()))[0]
            fasta_name = os.path.splitext(os.path.basename(self.fasta_path.get()))[0]
            suggested = os.path.join(os.path.dirname(self.fasta_path.get()), f"{fasta_name}__{model_name}_predictions.csv")
            self.output_path.set(suggested)

    # --- Logging helper (safe to call from either thread) ---

    def _log(self, message):
        self.log_queue.put(message)

    def _poll_log_queue(self):
        while not self.log_queue.empty():
            message = self.log_queue.get_nowait()
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.root.after(100, self._poll_log_queue)

    # --- Run button handler ---

    def _on_run_clicked(self):
        model_path = self.model_path.get().strip()
        fasta_path = self.fasta_path.get().strip()
        output_path = self.output_path.get().strip()
        use_cache = self.use_cache_var.get()
        cache_path = self.cache_load_path.get().strip()
        save_cache = self.save_cache_var.get()

        if not model_path or not fasta_path or not output_path:
            messagebox.showwarning("Missing input", "Please select a model file, a candidate FASTA file, and an output location before running.")
            return
        if use_cache and not cache_path:
            messagebox.showwarning("Missing input", "You checked 'Use previously saved embeddings' but haven't selected a cache file.")
            return

        self.run_button.config(state="disabled")
        self.progress_bar.start(12)
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        worker = threading.Thread(
            target=self._run_in_background,
            args=(model_path, fasta_path, output_path, self.device_choice.get(),
                  use_cache, cache_path, save_cache),
            daemon=True,
        )
        worker.start()

    def _run_in_background(self, model_path, fasta_path, output_path, device_choice,
                            use_cache, cache_path, save_cache):
        try:
            output_csv, n_flagged, n_total = run_prediction(
                model_path, fasta_path, output_path, self._log,
                device_choice=device_choice,
                use_cached_embeddings=use_cache,
                cached_embeddings_path=cache_path if use_cache else None,
                save_embeddings_after=save_cache,
            )
            self.root.after(0, lambda: self._on_finished(
                success=True,
                message=f"{n_flagged}/{n_total} candidates flagged as likely degraders.\nSaved to:\n{output_csv}"
            ))
        except Exception as e:
            self.log_queue.put(f"\nERROR: {e}")
            self.root.after(0, lambda: self._on_finished(success=False, message=str(e)))

    def _on_finished(self, success, message):
        self.progress_bar.stop()
        self.run_button.config(state="normal")
        if success:
            messagebox.showinfo("Done", message)
        else:
            messagebox.showerror("Something went wrong", message)


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
