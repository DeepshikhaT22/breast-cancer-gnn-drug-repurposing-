# breast-cancer-gnn-drug-repurposing-
# Breast Cancer Subtype-Specific Drug Repurposing with Heterogeneous Graph Neural Networks

This repository contains the code and data for our paper:  
**"A Heterogeneous Graph Neural Network Framework for Subtype-Specific Drug Efficacy Prediction in Breast Cancer"**  
Deepshikha Tripathi, Dr. Perugu Shyam

We integrate multi-omics data (gene expression, copy number variation, mutation) from breast cancer cell lines with protein-protein interactions, pathway annotations, and drug-target associations to build a heterogeneous graph. A corrected NeoDTI model (with mean aggregation and layer normalization) predicts drug efficacy and identifies subtype-specific drug candidates, with a focus on triple-negative breast cancer (TNBC).

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Graph Construction](#graph-construction)
  - [Model Training](#model-training)
  - [Evaluation and Analysis](#evaluation-and-analysis)
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

The best model (NeoDTI) achieves test ROC-AUC 0.767 and Precision@10 0.9. It identifies globally effective drugs (e.g., exatecan-mesylate, torin 2) and, importantly, uncovers TNBC-preferential drugs targeting ion channels and GPCRs—validated by pathway enrichment and literature.

## Repository Structure

breast-cancer-gnn-drug-repurposing/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/
│ ├── processed/ # Small processed files (node mappings, drug lists)
│ └── raw/ # Instructions to download raw data
├── notebooks/
│ ├── 01_data_preprocessing.ipynb
│ ├── 02_graph_construction.ipynb
│ ├── 03_neodti_training.ipynb
│ ├── 04_rgcn_training.ipynb
│ ├── 05_analysis_and_figures.ipynb
├── src/
│ ├── models.py
│ ├── data_utils.py
│ ├── graph_utils.py
│ └── train_utils.py
├── results/
│ ├── figures/ # All publication figures
│ ├── tables/ # Result tables (CSV)
│ └── models/ # Trained model weights (optional)
└── docs/ # Additional documentation
