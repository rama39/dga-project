# dga-project

Domain Generation Algorithm detection system using LSTM for CIN0114 - Técnicas de Ataque e Detecção de Intrusão - T01 (2026.1) CIN UFPE.

This repository contains the implementation files and dataset resources for a DGA detection experiment inspired by the paper: https://arxiv.org/pdf/1611.007091.

## Contents

- `project.ipynb` - Jupyter notebook for model exploration and evaluation
- `dga_project/` - Python package with dataset loading and classifier implementations
- `dga_project/datasets/` - Sample domain data files used by the project

## Running in Google Colab

To run the project in Google Colab, follow these steps:

1. Download `project.ipynb` and the `dga_project/` folder into the same base folder in your Google Drive.
2. Open `project.ipynb` in Google Colab.
3. Allow any requested permissions so Colab can access files from your Google Drive.
4. Run the notebook cells in order.

> Note: The notebook assumes the `dga_project/` folder is available in the notebook working directory or mounted Google Drive folder.

## Notes

- If the notebook references local dataset files, make sure the `dga_project/datasets/` folder is present alongside `project.ipynb`.
- The implementation includes support for LSTM-based DGA classification and additional classifiers under `dga_project/dga_classifier/`.
- If you store the project in a different Drive folder, update PROJECT_DIR in the first cell accordingly
