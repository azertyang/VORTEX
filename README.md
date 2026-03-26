# VORTEX - PULL BEFORE PULLING
### **V**ariational **O**mics **R**epresentation via **T**issue-aware **E**mbedding and **X**-modal learning
 
> Self-supervised spatial GNN for multi-modal transcriptomics representation learning  
> Deep Learning Project — University of Bern, 2026
 
---
 
## Overview
 
VORTEX is a graph-based self-supervised model that learns spatially-aware cell embeddings from CosMx spatial transcriptomics data. Instead of treating each cell as a flat vector of gene counts, VORTEX models cells as nodes in a spatial graph — where edges connect physically proximate cells in tissue.
 
The model is pre-trained via **masked gene prediction**: a fraction of gene counts is masked per cell, and the model reconstructs them using spatial neighborhood context.
 
VORTEX builds on and extends [MISO](https://github.com/kpcoleman/miso) (Coleman et al., Nature Methods 2025).
 
---
 
## Key Features
 
- **GATv2-based encoder** with physical distance as edge features
- **Masked gene prediction** as self-supervised pretraining task
- **Subcellular localization** (nucleus vs. membrane zone) as node features — unique to CosMx
- **No H&E image required** — works purely from molecular measurements
- Evaluated on CosMx Mouse Brain and NSCLC data
 
---
 
## Architecture
 
```
CosMx data (gene counts + spatial coords + subcellular zone)
        ↓
  Spatial Graph Construction (k-NN on coordinates)
        ↓
  GATv2 Encoder (edge features = physical distance)
        ↓
  Latent Embedding z ∈ R^128
        ↓
  MLP Decoder → reconstruct masked genes
```
 
---
 
## Installation
 
```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/VORTEX.git
cd VORTEX
 
# 2. Create conda environment
conda create -n vortex python=3.9
conda activate vortex
 
# 3. Install dependencies
pip install torch torchvision
pip install torch_geometric
pip install scanpy anndata
pip install numpy pandas matplotlib seaborn scikit-learn
```
 
> **Note**: VORTEX does not require the ViT histology checkpoint from MISO.  
> If cloning MISO as a submodule, skip the LFS checkpoint with `GIT_LFS_SKIP_SMUDGE=1 git clone`.
 
---
 
## Data
 
We use the **CosMx Mouse Brain FFPE dataset** (publicly available):
- 🔗 https://bruker.com/spatial-biology/cosmx-smi/ffpe-dataset/mouse-brain
 
```bash
# Structure expected
data/
├── mouse_brain/
│   ├── counts.h5ad       # gene expression AnnData
│   └── metadata.csv      # cell coordinates + subcellular zone
└── nsclc/
    └── counts.h5ad       # for zero-shot transfer experiment
```
 
---
 
## Usage
 
```bash
# Run pretraining with masked gene prediction
python train.py \
    --data data/mouse_brain/counts.h5ad \
    --mask_ratio 0.15 \
    --n_layers 3 \
    --hidden_dim 128 \
    --latent_dim 64 \
    --epochs 100
 
# Extract embeddings
python embed.py \
    --checkpoint checkpoints/vortex_best.pt \
    --data data/mouse_brain/counts.h5ad \
    --output embeddings/mouse_brain.npy
```
 
---
 
## Experiments
 
| # | Experiment | Description |
|---|-----------|-------------|
| 1 | **Spatial ablation** | Clustering with vs. without spatial edges (ARI metric) |
| 2 | **Linear probe** | Do embeddings capture subcellular gene localization? |
| 3 | **Zero-shot transfer** | Mouse Brain → NSCLC, no retraining |
 
```bash
# Experiment 1 — ablation
python experiments/ablation_spatial.py
 
# Experiment 2 — linear probe
python experiments/linear_probe.py
 
# Experiment 3 — zero-shot transfer
python experiments/zero_shot_transfer.py \
    --source mouse_brain --target nsclc
```
 
---
 
## Baselines
 
We compare VORTEX against:
- **MISO** (Coleman et al., 2025)
- **GraphST** (He et al., 2023)
- **No-spatial baseline** (same architecture, edges removed)
 
---
 
## Project Structure
 
```
VORTEX/
├── data/                  # datasets (not tracked)
├── vortex/
│   ├── model.py           # GATv2 encoder + MLP decoder
│   ├── dataset.py         # CosMx data loader + graph construction
│   ├── masking.py         # masked gene prediction logic
│   └── train.py           # training loop
├── experiments/
│   ├── ablation_spatial.py
│   ├── linear_probe.py
│   └── zero_shot_transfer.py
├── notebooks/
│   └── exploration.ipynb  # data exploration
├── checkpoints/           # saved models
├── train.py               # main entry point
├── embed.py               # embedding extraction
├── requirements.txt
└── README.md
```
 
---
 
## Team
 
- Mathilde Jacquey
- Aurélien Mroczek  
- Florence Kurz
 
*University of Bern — Deep Learning Course 2026*
 
---
 
## References
 
- Coleman et al. *Resolving tissue complexity by multimodal spatial omics modeling with MISO.* Nature Methods, 2025.
- He et al. *Spatially informed clustering, integration, and deconvolution of spatial transcriptomics with GraphST.* Nature Communications, 2023.
- Bravo Gonzalez-Blas et al. *SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks.* Nature Methods, 2023.
 
