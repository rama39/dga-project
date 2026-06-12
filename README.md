# dga-project

Domain Generation Algorithm (DGA) detection system using an LSTM network, developed for **CIN0114 — Técnicas de Ataque e Detecção de Intrusão — T01 (2026.1)**, CIn UFPE.

This repository reproduces and extends the DGA detection experiments from the paper *Predicting Domain Generation Algorithms with Long Short-Term Memory Networks* (Woodbridge et al., 2016): https://arxiv.org/pdf/1611.00791.

The work is organized in two phases:

- **Phase 1** — reproduction on the paper's original data sources: Alexa Top 1M (benign) and the Bambenek OSINT DGA feed (malicious).
- **Phase 2** — re-running the same system on different datasets: **Cisco Umbrella Top 1M** (benign) and **DGArchive** `2016-09-19-dgarchive_full.tgz` (malicious), with a preprocessing step that converts these sources into the exact input format the pipeline expects.

## Repository structure

```
.
├── colabs/
│   ├── Projeto_fase1.ipynb        # Phase 1 notebook (Alexa + Bambenek)
│   └── Projeto_fase2.ipynb        # Phase 2 notebook (Umbrella + DGArchive)
├── projeto/
│   ├── datasets/
│   │   ├── bambenek_dga_domain_30.csv   # malicious input: DGA_family,Domain,Type
│   │   └── top-1m.csv                    # benign input: rank,domain (no header)
│   ├── dga_classifier/
│   │   ├── __init__.py
│   │   ├── data.py                # data loading / input contract
│   │   ├── lstm.py                # LSTM classifier (binary + multiclass)
│   │   ├── bigram.py              # logistic regression on character bigrams
│   │   ├── randomforest.py        # random forest on manual features
│   │   └── hmm.py                 # HMM baseline
│   └── run.py                     # runs the experiments and produces the ROC figure
├── resultados/
│   ├── results_roc_fase1.png      # ROC curves — Phase 1
│   └── results_roc_fase2.png      # ROC curves — Phase 2
├── .gitignore
└── README.md
```

## Input data contract

Both phases share the same pipeline, which reads exactly two files from `projeto/datasets/`:

- `bambenek_dga_domain_30.csv` — columns `DGA_family,Domain,Type` (malicious domains; `Type` = `DGA`).
- `top-1m.csv` — `rank,domain` with **no header** (benign domains).

`projeto/dga_classifier/data.py` loads both, strips the top-level domain, and emits records of the form `(label, domain, family)`, where `label ∈ {benign, dga}` and `family` is a canonical name shared with the multiclass, super-family, leave-class-out and HMM experiments.

## Running in Google Colab

1. Upload the `projeto/` folder (and, for Phase 2, the DGArchive `.tgz`) to your Google Drive, or clone this repository.
2. Open the desired notebook from `colabs/` in Google Colab:
   - `Projeto_fase1.ipynb` for the original datasets;
   - `Projeto_fase2.ipynb` for the Umbrella + DGArchive datasets.
3. Enable a GPU runtime: *Runtime → Change runtime type → GPU*.
4. Grant the requested permissions so Colab can mount your Google Drive.
5. Set `PROJECT_DIR` in the setup cell to point at your `projeto/` folder, then run the cells in order.

### Phase 2 specifics

The Phase 2 notebook handles the dataset adaptation end to end:

- downloads the Cisco Umbrella list directly (`top-1m.csv.zip`);
- reads the DGArchive `2016-09-19-dgarchive_full.tgz` by streaming (one CSV per family);
- maps DGArchive family names to the paper's canonical families, deduplicates, removes benign/DGA leakage, and writes `bambenek_dga_domain_30.csv` and `top-1m.csv` into `projeto/datasets/

  Dataset (DGArchive)
The `2016-09-19-dgarchive_full.tgz` file is not versioned (637 MB, restricted access).
Download it via Google Drive: (https://drive.google.com/file/d/14YrtWRkW7PQ1hZNtz3nAfIcaqGwoK60E/view?usp=sharing)

## Experiments

`projeto/run.py` reproduces the three experimental designs from the paper:

1. **Binary classification** (DGA vs. non-DGA), compared against LSTM, character-bigram logistic regression, manual-feature random forest, and HMM baselines.
2. **Leave-class-out** binary classification, to measure robustness to DGA families unseen during training.
3. **Multiclass / super-family** classification, to attribute a domain to a specific DGA family.

Execution is controlled through environment variables (e.g. `DGA_MAX_PER_FAMILY`, `DGA_MAX_BENIGN`, `DGA_NFOLDS`, `DGA_RUN_HMM`, `DGA_ROC_MODE`), which let the experiments fit within Colab's time and memory limits. The resulting ROC figures are saved under `resultados/`.

## Notes

- Make sure `projeto/datasets/` exists alongside `projeto/run.py` before running; the Phase 2 notebook (re)creates the two CSV files in that folder.
- Cisco Umbrella ranks domains by DNS query volume rather than web traffic (as the now-retired Alexa list did). The CSV format is identical, so it is a drop-in replacement, but absolute false-positive rates may differ from the original paper.
- If you store the project in a different Drive folder, update `PROJECT_DIR` in the first notebook cell accordingly.
