# Breast Cancer Subtype‑Specific Drug Repurposing with Heterogeneous Graph Neural Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) (add after publication)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the code and data for our paper:  
**"A Heterogeneous Graph Neural Network Framework for Subtype-Specific Drug Efficacy Prediction in Breast Cancer"**  
Deepshikha Tripathi, Dr. Perugu Shyam

We integrate multi‑omics data (gene expression, copy number variation, mutation) from breast cancer cell lines with protein‑protein interactions, pathway annotations, and drug‑target associations to build a heterogeneous graph. A corrected NeoDTI model (with mean aggregation and layer normalization) predicts drug efficacy and identifies subtype‑specific drug candidates, with a focus on triple‑negative breast cancer (TNBC).

## Table of Contents
- [Overview](#overview)
- [Interactive Web App](#interactive-web-app)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Results](#results)
- [Reproducing the Paper](#reproducing-the-paper)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Overview

We constructed a heterogeneous graph with:
- **Nodes**: Drugs (4,594), Genes (19,609), Pathways (2,351), Cell lines (51)
- **Edges**: Drug–gene, gene–gene (PPI), gene–pathway, cell–gene (overexpression, amplification, mutation)
- **Reverse edges** added for bidirectional flow

We trained two GNN models:
- **RGCN** (Relational Graph Convolutional Network)
- **Corrected NeoDTI** (with mean aggregation and layer norm)

The best model (NeoDTI) achieves test ROC‑AUC 0.767 and Precision@10 0.9. It identifies globally effective drugs (e.g., exatecan‑mesylate, torin 2) and, importantly, uncovers TNBC‑preferential drugs targeting ion channels and GPCRs – validated by pathway enrichment and literature.

## Interactive Web App

A **Streamlit web application** is available to explore the model predictions interactively:

- **Explore by Subtype** – View top‑N drugs for HER2, Luminal, and TNBC, filter by target class, and download full rankings.
- **Search by Gene** – Find all drugs targeting a specific gene, with predicted scores per subtype.
- **Compare Drugs** – Compare two drugs side by side with bar charts, shared target genes, and common pathways.
- **Visualize Drug Network** – Interactive network of a drug, its target genes, and associated pathways.
- **Predict for New Cell Line** – Upload a CSV with expression Z‑scores, CNV log2 ratios, and mutation status to get predicted drug rankings for a custom cell line.

### Run the app locally

1. **Clone this repository**.
2. **Create a conda environment** (recommended) or use a virtual environment with Python 3.9+.
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
