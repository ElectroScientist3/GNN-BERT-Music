# GNN–BERT Music Context Understanding

This repository contains the implementation of the GNN-BERT framework for Music Context Understanding. The project focuses on learning rich multimodal representations of music by combining structural acoustic context (using Graph Neural Networks) with textual/contextual metadata (using BERT).

## Overview

The project tackles the following 4 core tasks:
1. **Dynamic Emotion Prediction**: Predicting continuous valence and arousal values over time.
2. **Music Tagging (Multi-label Classification)**: Predicting mood and genre tags.
3. **Cross-Modal Retrieval (Contrastive Learning)**: Learning a joint embedding space to retrieve audio given text and vice versa.
4. **Music Context Summarization**: Generating rich, context-aware summaries of a track's audio and metadata.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd gnn-bert-music-context
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows use:
   venv\Scripts\activate
   # On Linux/Mac use:
   # source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

We use the DEAM (MediaEval Database for Emotional Analysis in Music) dataset.
1. Download the dataset archives (`annotations.zip`, `features.zip`, `audio.zip`).
2. Extract the DEAM zips into the raw data directory:
   ```
   data/raw/
   ├── annotations/
   ├── features/
   └── audio/
   ```

## Usage

### 1. Preprocessing
Run the preprocessing script to parse annotations, extract audio features, construct the graph, and generate text features:
```bash
python scripts/preprocess.py --config config.yaml
```

### 2. Training
Train the models for the respective tasks:

- **Emotion Prediction:**
  ```bash
  python train.py --task emotion --config config.yaml
  ```
- **Music Tagging:**
  ```bash
  python train.py --task tagging --config config.yaml
  ```
- **Cross-Modal Retrieval:**
  ```bash
  python train.py --task retrieval --config config.yaml
  ```
- **Summarization:**
  ```bash
  python train.py --task summarization --config config.yaml
  ```

### 3. Evaluation
Evaluate the trained models:
```bash
python evaluate.py --task all --checkpoint_dir checkpoints/ --config config.yaml
```

## Project Structure

```text
gnn-bert-music-context/
├── data/                   # Data directories (raw, processed, splits)
├── models/                 # Model definitions (GNN, BERT, Fusion)
├── scripts/                # Preprocessing and utility scripts
├── train.py                # Main training script
├── evaluate.py             # Evaluation script
├── config.yaml             # Central configuration
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Results

| Task | Metric | Value |
|------|--------|-------|
| Emotion Prediction | RMSE | TBD |
| Music Tagging | mAP | TBD |
| Retrieval | R@10 | TBD |
| Summarization | ROUGE-L | TBD |

## References

- DEAM: A Dataset for Emotion Analysis in Music
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- GraphSAGE: Inductive Representation Learning on Large Graphs

## Authors

- Your Name / Organization
