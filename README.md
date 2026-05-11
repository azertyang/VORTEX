# VORTEX 
## **V**ariational **O**mics **R**epresentation via **T**issue-aware **E**mbedding and **X**-modal learning
> Project for the Deep Learning course — University of Bern. Presentation date: May 11, 2026.

---
**Parameter-efficient fine-tuning (PEFT) of Nicheformer** ([Tejada-Lapuerta, Schaar et al.,2025](https://doi.org/10.1038/s41592-025-02814-z)) **using LoRA** [(Hu et al., 2022)](https://doi.org/10.48550/arXiv.2106.09685) **for cell-type classification in spatial transcriptomics, featuring a cross-technology generalization study (CosMx → MERFISH)**.

---
 
## Team
 
- Mathilde Jacquey
- Florence Kurz
- Aurélien Mroczek
 
*University of Bern — Deep Learning Course 2026*

---

### Context
The first version of this project followed a linear probing setup: a logistic regression was trained on top of frozen Nicheformer embeddings. The course feedback flagged this as a weakness. 

This repository addresses that. We add real trainable parameters inside the transformer, LoRA adapters injected into every attention layer, while staying within the constraints (single GPU, no full pre-training, no end-to-end fine-tuning of the 90M+ parameter backbone). Only ~1.3% of weights are trained.

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
                                     logits over 8 cell types
```

According to LoRA (Hu et al. 2022): 
For each attention projection W, we learn two low-rank matrices A∈R (r×d) and B∈R(d×r) with r≪d. The forward becomes:
W' x = W⋅x with (W+ r/α BA)⋅x
At initialization, B=0, so the model behaves exactly like the pre-trained version. Only A and B are updated.

### Unified label space (8 classes):

CosMx provides 49 subtypes, e.g., Excitatory.neurons.layer.4, MERFISH provides 16 broader classes. To make cross-technology comparison meaningful, we map both to 8 major cell types:

| Class | CosMx subtypes (49 → 8) | MERFISH (16 → 8) |
|---|---|---|
| Astrocyte | 2 subtypes | `Astrocyte` |
| Excitatory | 20 subtypes | `Excitatory` |
| Inhibitory | 10 subtypes | `Inhibitory` |
| Oligodendrocyte | 5 subtypes | `OD Mature/Immature 1-4` |
| Microglia | `Microglia`, `Perivascular.macrophages`, `T.cell` | `Microglia` |
| Vascular | `Pericytes`, `Endothelial`, `Leptomeningeal`, `Smooth muscle` | `Endothelial 1-3`, `Pericytes` |
| Ependymal | `Ependymal`, `Hypendymal`, `Tanycytes` | `Ependymal` |
| Other | `Choroid`, `Neuroblasts`, `Radial.glia` | `Ambiguous` |

### Training tricks

The naive setup falls into a local minimum where the model predicts the majority class (Excitatory, 41% of the CosMx dataset). We adress this with:
- class-weighted cross-entropy (normalized to mean 1)
- label smoothing = 0.1 to reduce overconfidence
- separate learning rates - lr=5e-4 for LoRA pre-trained and lr=5e-3 for the randomly-initialized MLP head
- fixed macro_F1 ; we count F1=0 for never-predicted classes (instead of skipping them)

### Repository Structure

```
nicheformer-lora/
├── README.md                       ← You are here
├── requirements.txt
├── LICENSE
├── scripts/
│   └── nicheformer_main.py      
├── slurm/
│   └── run_nicheformer_main.slurm       
└── data/                           
    ├── pretrained/                 ← Checkpoints + model.h5ad + means
    ├── raw/                        ← Raw AnnData CosMx + MERFISH 
    (empty, available on demand)
    ├── outputs/                        ← Raw AnnData CosMx + MERFISH
    |   ├── metrics.csv                
    |   ├── summary.txt    
    |   └── figures/                   
    └── processed/                  ← Prepared AnnData
```
---

### Reproducing the Project on Ubelix
#### 0. Prerequisites

UniBe account with Ubelix access (ssh <user>@submit.unibe.ch).

Access to the gpu partition (1 GPU, either RTX4090, A100 or H100).

Takes about 1 to 2 hours for 15 epochs

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
If Mendeley link is broken, contact authors directly.

Create the following directory tree:
- pretrained/nicheformer.ckpt
- raw/cosmx_mouse_brain.h5ad and raw/merfish_mouse_brain.h5ad



#### 3. Download Spatial Data

[CosMx Mouse Brain](https://doi.org/10.1038/s41587-022-01483-z): (~45k cells) via Bruker/Nanostring portal. Save to data/raw/cosmx_mouse_brain.h5ad.

[MERFISH Mouse Brain](https://doi.org/10.1038/s41586-023-06808-9): (~73k cells) via squidpy:

```
import squidpy as sq
adata = sq.datasets.merfish()
adata.write_h5ad('data/raw/merfish_mouse_brain.h5ad')
```
More details on [squidpy](https://squidpy.readthedocs.io/en/stable/), python package, [available article linked to this package](https://doi.org/10.1038/s41592-021-01358-2). 

#### 4. Run

```
conda activate nicheformer
python scripts/nicheformer_main.py
```

#### 5. Check the outputs

The directory tree must be similar to this one below:

```
outputs/
├── checkpoints/
│   ├── best-XX-0.YYY.ckpt
│   └── last.ckpt
├── logs/version_0/metrics.csv      # full training trace
├── preds_indomain.npy
├── labels_indomain.npy
├── preds_crosstech.npy
├── labels_crosstech.npy
├── summary.txt
└── figures/
    ├── train_curves.png
    ├── comparison.png
    ├── confusion_indomain.png
    └── confusion_crosstech.png
```

---

### Key Hyperparameters
Parameter	Default	Justification
- lora_r	8	Good quality/cost trade-off (Hu et al. 2022)
- lora_alpha	16	Standard 2×r convention
- lora_target_layers all(12)
- lr (LoRA params)	5e-4	
- lr (head params)  5e-3
- warmup_steps 50
- batch size 16
- precision	16-mixed (Halves activation memory)
- weight decay 0.01

### Results
In-domain (CosMx test): F1 macro > 0.80 (linear probing reaches ~0.75).

Cross-tech (MERFISH): Expected drop F1 macro < 0.1 (77.3 points). 

Distribution of In-domain predictions (see also: `data/outputs/run_fix/summary.txt`):
- astrocyte: 715
- excitatory: 2,678
- inhibitory: 1,421
- oligodendrocyte: 1,046
- microglia: 229
- vascular: 700
- ependymal: 134
- other: 104

Distribution of Cross-tech predictions (see also: `data/outputs/run_fix/summary.txt`):
- oligodendrocyte: 28,339
- microglia: 28,375
- vascular: 16,941
  
--- 

## Discussion

The 77-point F1 drop between CosMx and MERFISH is striking and warrants analysis. Three non-exclusive hypotheses:

1. **Vocabulary mismatch.** MERFISH measures only 161 genes vs CosMx's 950. After alignment to Nicheformer's 20,310-gene vocabulary, the MERFISH input is much sparser, and the per-gene technology-mean normalization shifts the token-rank distribution.
   
2. **Annotation taxonomy.** Even after our 8-class harmonization, "Inhibitory" in MERFISH encompasses a different set of subtypes than "Inhibitory" in CosMx. The decision boundary learned on CosMx is too sharp for the coarser MERFISH labels.

3. **LoRA over-specialization.** Hu et al. (2022) observed that LoRA can over-fit to the source domain when the rank is small relative to the domain shift. r = 8 may be too restrictive for cross-technology transfer; higher ranks or zero-shot evaluation could mitigate this.

A natural extension would be a **domain-adversarial head** that explicitly pushes CosMx and MERFISH embeddings into a shared space during training.

 
---
### Related work 
The repository [github direction](https://github.com/nnayz/ft-nicheformer), explores Nicheformer fine-tuning in a federated learning setting. We were aware of it but did not use any of its code; our approach (LoRA + cross-tech evaluation) is independent.

### References
- Tejada-Lapuerta, A., Schaar, A.C., Gutgesell, R. et al. Nicheformer: a foundation model for single-cell and spatial omics. Nat Methods 22, 2525–2538 (2025). https://doi.org/10.1038/s41592-025-02814-z
- Edward J. Hu et al: LoRA: Low-Rank Adaptation of Large Language Models (2021). https://doi.org/10.48550/arXiv.2106.09685
- He, S., Bhatt, R., Brown, C. et al. High-plex imaging of RNA and proteins at subcellular resolution in fixed tissue by spatial molecular imaging. Nat Biotechnol 40, 1794–1806 (2022). https://doi.org/10.1038/s41587-022-01483-z
- Zhang, M., Pan, X., Jung, W. et al. Molecularly defined and spatially resolved cell atlas of the whole mouse brain. Nature 624, 343–354 (2023). https://doi.org/10.1038/s41586-023-06808-9
- Palla, G., Spitzer, H., Klein, M. et al. Squidpy: a scalable framework for spatial omics analysis. Nat Methods 19, 171–178 (2022). https://doi.org/10.1038/s41592-021-01358-2
 
