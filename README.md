# VORTEX 
## **V**ariational **O**mics **R**epresentation via **T**issue-aware **E**mbedding and **X**-modal learning
> Project for the Deep Learning course — University of Bern. Presentation date: May 11, 2026.

---
 
## Team
 
- Mathilde Jacquey
- Florence Kurz
- Aurélien Mroczek
 
*University of Bern — Deep Learning Course 2026*

---
**Parameter-efficient fine-tuning (PEFT) of Nicheformer** ([Tejada-Lapuerta, Schaar et al., Nature Methods 2025](https://doi.org/10.1038/s41592-025-02814-z)) **using LoRA** [(Hu et al., 2022)](https://doi.org/10.48550/arXiv.2106.09685) **for cell-type classification in spatial transcriptomics, featuring a cross-technology generalization study (CosMx → MERFISH)**.

Convex work available at this [github direction](https://github.com/nnayz/ft-nicheformer), which used federated learning framework to fine-tuned Nicheformer on spatial single-cell transcriptomics data, originating from mouse brain.


### Context
The initial evaluation feedback pointed out a weakness in the "linear probing" version of the project:

"Pure linear probing — no gradient updates to any neural network during the project. The 'DL' work is borrowed entirely from Nicheformer; your team only trains logistic regression on top."

This repository solves that issue. We add real trainable parameters inside the transformer (LoRA adapters) while staying within the computational constraints (single GPU, no pre-training from scratch). The 90M+ parameter backbone remains frozen; we only train ~0.3% of the weights.

### Architecture

```
   gene expression tokens ───▶  Nicheformer
                                ┌─────────────────────────┐
                                │  12-layer Transformer   │
                                │  (Frozen)               │
                                │   ┌───────────────┐     │
                                │   │ MHA + LoRA Δ  │ ◀── Only weights trained
                                │   │ (Q, K, V, out)│     │
                                │   └───────────────┘     │
                                │   ┌───────────────┐     │
                                │   │ FFN (Frozen)  │     │
                                │   └───────────────┘     │
                                └─────────────────────────┘
                                             │
                                        mean pooling
                                             │
                                   ┌─────────────────┐
                                   │ MLP head        │ ◀── Trained
                                   │ Linear-Tanh-Drop│
                                   │ -Linear         │
                                   └─────────────────┘
                                             │
                                     logits over N types
```

According to LoRA (Hu et al. 2022): For each attention projection W, we learn two low-dimension matrices A∈R r×d and B∈R d×r with r≪d. We replace W⋅x with (W+ r/α BA)⋅x. At initialization, B=0, so the model behaves exactly like the pre-trained version.

### Repository Structure

```
nicheformer-lora/
├── README.md                       ← You are here
├── requirements.txt
├── configs/
│   ├── data_prep.yaml              ← Data prep config
│   └── train_cosmx.yaml            ← Training config
├── src/
│   ├── lora.py                     ← LoRA module (self-contained, ~250 lines)
│   ├── model.py                    ← LightningModule for FT + classification
│   └── data.py                     ← Vocab alignment + context tokens + split
├── scripts/
│   ├── setup_ubelix_env.sh         ← Create conda env on Ubelix
│   ├── prepare_data.py             ← Prepares CosMx + MERFISH
│   ├── train.py                    ← Training script
│   ├── evaluate_cross_tech.py      ← Eval CosMx-trained → MERFISH
│   └── download_pretrained.py      ← Mendeley download helper
├── slurm/
│   ├── train.slurm                 ← SLURM job for training (GPU)
│   └── eval_cross_tech.slurm       ← SLURM job for evaluation
├── notebooks/
│   └── demo.ipynb                  ← Demo notebook for presentation
├── tests/
│   ├── test_lora.py                ← LoRA unit tests (pass in <5s)
│   └── test_integration.py         ← Integration test on mini-transformer
└── data/                           ← Gitignored
    ├── pretrained/                 ← Checkpoints + model.h5ad + means
    ├── raw/                        ← Raw AnnData CosMx + MERFISH
    └── processed/                  ← Prepared AnnData
```
---

### Reproducing the Project on Ubelix
#### 0. Prerequisites

UniBe account with Ubelix access (ssh <user>@submit.unibe.ch).

Access to the gpu partition.

~2 GB free quota in $HOME for the environment + ~5 GB for checkpoints.

#### 1. Clone and Setup

```
ssh <user>@submit.unibe.ch
cd $HOME
git clone https://github.com/<your-username>/nicheformer-lora.git
cd nicheformer-lora
bash scripts/setup_ubelix_env.sh    # Creates nicheformer_env (~5–10 min)
```

#### 2. Download Nicheformer Resources

The authors distribute weights on Mendeley Data. Download these files locally and scp them to data/pretrained/:

[model.h5ad, cosmx_mean_script.npy, merfish_mean_script.npy](https://github.com/theislab/nicheformer/tree/main/data/model_means), [nicheformer.ckpt](https://data.mendeley.com/preview/87gm9hrgm8?a=d95a6dde-e054-4245-a7eb-0522d6ea7dff).



#### 3. Download Spatial Data

[CosMx Mouse Brain](https://doi.org/10.1038/s41587-022-01483-z): (~45k cells) via Bruker/Nanostring portal. Save to data/raw/cosmx_mouse_brain.h5ad.

[MERFISH Mouse Brain](https://doi.org/10.1038/s41586-023-06808-9): (~73k cells) via squidpy:

```
import squidpy as sq
adata = sq.datasets.merfish()
adata.write_h5ad('data/raw/merfish_mouse_brain.h5ad')
```
More details on [squidpy](https://squidpy.readthedocs.io/en/stable/), python package, [available article linked to this package](https://doi.org/10.1038/s41592-021-01358-2). 

#### 4. Prepare Datasets

```
conda activate nicheformer_env
python scripts/prepare_data.py --config configs/data_prep.yaml
```

#### 5. Launch GPU Training

```
sbatch slurm/train.slurm
Monitor progress: tail -f logs/slurm-nicheformer-lora-<JOBID>.out.
Outputs (checkpoints and LoRA weights) will be saved in outputs/.
```

#### 6. Cross-Technology Evaluation

```
sbatch --export=ALL,RUN_DIR=outputs/cosmx_lora_r8_<timestamp> \
       slurm/eval_cross_tech.slurm
```

---

### Key Hyperparameters
Parameter	Default	Justification
- lora_r	8	Good quality/cost trade-off (Hu et al. 2022)
- lora_alpha	16	Standard 2×r convention
- lora_target_layers	All (12)	Max expressivity
- lr	5e-4	Higher than full-FT due to fewer parameters
- precision	16-mixed	Halves activation memory 

### Expected Results
In-domain (CosMx test): F1 macro > 0.80 (linear probing reaches ~0.75).

Cross-tech (MERFISH): Expected drop of 15–30 F1 points. The scientific question: Does LoRA help generalization or does it over-specialize the model?

Rank Ablation: Repeat training with r∈{2,4,8,16,32} and plot F1 vs r.

 
---
 
## References
- Tejada-Lapuerta, A., Schaar, A.C., Gutgesell, R. et al. Nicheformer: a foundation model for single-cell and spatial omics. Nat Methods 22, 2525–2538 (2025). https://doi.org/10.1038/s41592-025-02814-z
- Edward J. Hu et al: LoRA: Low-Rank Adaptation of Large Language Models (2021). https://doi.org/10.48550/arXiv.2106.09685
- He, S., Bhatt, R., Brown, C. et al. High-plex imaging of RNA and proteins at subcellular resolution in fixed tissue by spatial molecular imaging. Nat Biotechnol 40, 1794–1806 (2022). https://doi.org/10.1038/s41587-022-01483-z
- Zhang, M., Pan, X., Jung, W. et al. Molecularly defined and spatially resolved cell atlas of the whole mouse brain. Nature 624, 343–354 (2023). https://doi.org/10.1038/s41586-023-06808-9
- Palla, G., Spitzer, H., Klein, M. et al. Squidpy: a scalable framework for spatial omics analysis. Nat Methods 19, 171–178 (2022). https://doi.org/10.1038/s41592-021-01358-2
 
