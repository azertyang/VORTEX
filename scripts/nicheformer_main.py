"""
Nicheformer + LoRA — VERSION FIX (lr augmenté + class weights + macro_f1 corrigé)

Corrige les bugs détectés sur la version précédente :
  1. macro_f1 ne skip plus les classes "jamais prédites" → F1 honnête
  2. CrossEntropyLoss avec class_weights pour éviter le shortcut "tout Excitatory"
  3. lr 5e-4 au lieu de 1e-4 + accumulate=2 pour des gradient updates plus fréquents

Usage : python script_fix.py
"""

from pathlib import Path
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import anndata as ad
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import optim
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

from nicheformer.data.dataset import NicheformerDataset
from nicheformer.models import Nicheformer
from nicheformer.models._utils import complete_masking


# =============================================================================
# CHEMINS
# =============================================================================
PROJECT_ROOT   = Path.home() / 'DeepLearning' / 'deep_learning_project_c' / 'data'
PRETRAINED_DIR = PROJECT_ROOT / 'pretrained'
RAW_DIR        = PROJECT_ROOT / 'raw'
PROCESSED_DIR  = PROJECT_ROOT / 'processed'
OUTPUT_DIR     = PROJECT_ROOT / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MAPPING UNIFIÉ — 8 classes
# =============================================================================
UNIFIED_CLASSES = {
    'Astrocyte': 0, 'Excitatory': 1, 'Inhibitory': 2, 'Oligodendrocyte': 3,
    'Microglia': 4, 'Vascular': 5, 'Ependymal': 6, 'Other': 7,
}

COSMX_TO_UNIFIED = {
    'Astrocytes.cortex.hippocampus': 'Astrocyte', 'Astrocytes.thalamus.hypothalamus': 'Astrocyte',
    'Excitatory.neurons.amygdala': 'Excitatory', 'Excitatory.neurons.di.mesencephalon': 'Excitatory',
    'Excitatory.neurons.hippocampal.CA1': 'Excitatory', 'Excitatory.neurons.hippocampal.CA2': 'Excitatory',
    'Excitatory.neurons.hippocampal.CA3': 'Excitatory', 'Excitatory.neurons.layer.1.piriform': 'Excitatory',
    'Excitatory.neurons.layer.2.3': 'Excitatory', 'Excitatory.neurons.layer.4': 'Excitatory',
    'Excitatory.neurons.layer.5': 'Excitatory', 'Excitatory.neurons.layer.5.6': 'Excitatory',
    'Excitatory.neurons.layer.6': 'Excitatory', 'Excitatory.neurons.telencephalon': 'Excitatory',
    'Granule.neurons': 'Excitatory', 'D1.medium.spiny.neurons': 'Excitatory',
    'D2.medium.spiny.neurons': 'Excitatory', 'Cholinergic.neurons.habenula': 'Excitatory',
    'Peptidergic.neurons': 'Excitatory', 'Serotonergic.neurons': 'Excitatory',
    'CCK.interneurons': 'Inhibitory', 'Hindbrain.inhibitory.neurons': 'Inhibitory',
    'Inhibitory.interneurons': 'Inhibitory', 'Inhibitory.neurons.amygdala': 'Inhibitory',
    'Inhibitory.neurons.habenula.hypothalamus': 'Inhibitory',
    'Inhibitory.neurons.habenula.thalamus': 'Inhibitory',
    'Inhibitory.neurons.reticular.nucleus': 'Inhibitory',
    'Interneuron.selective.interneurons': 'Inhibitory',
    'Neurogliaform.cells': 'Inhibitory', 'Telencephalon.inhibitory.neurons': 'Inhibitory',
    'Committed.oligodendrocytes': 'Oligodendrocyte', 'Mature.oligodendrocytes': 'Oligodendrocyte',
    'Myelin.forming.oligodendrocytes': 'Oligodendrocyte',
    'Newly.formed.oligodendrocytes': 'Oligodendrocyte',
    'Oligodendrocyte.precursor.cells': 'Oligodendrocyte',
    'Microglia': 'Microglia', 'Perivascular.macrophages': 'Microglia', 'T.cell': 'Microglia',
    'Pericytes': 'Vascular', 'Vascular.endothelial.cells': 'Vascular',
    'Vascular.leptomeningeal.cells': 'Vascular', 'Vascular.smooth.muscle.cells': 'Vascular',
    'Ependymal.cells': 'Ependymal', 'Hypendymal': 'Ependymal', 'Tanycytes': 'Ependymal',
    'Choroid.plexus.epithelial.cells': 'Other', 'Neuroblasts': 'Other',
    'Olfactory.ensheathing.cells': 'Other', 'Radial.glia': 'Other',
}

MERFISH_TO_UNIFIED = {
    'Astrocyte': 'Astrocyte', 'Excitatory': 'Excitatory', 'Inhibitory': 'Inhibitory',
    'OD Mature 1': 'Oligodendrocyte', 'OD Mature 2': 'Oligodendrocyte',
    'OD Mature 3': 'Oligodendrocyte', 'OD Mature 4': 'Oligodendrocyte',
    'OD Immature 1': 'Oligodendrocyte', 'OD Immature 2': 'Oligodendrocyte',
    'Microglia': 'Microglia', 'Endothelial 1': 'Vascular',
    'Endothelial 2': 'Vascular', 'Endothelial 3': 'Vascular', 'Pericytes': 'Vascular',
    'Ependymal': 'Ependymal', 'Ambiguous': 'Other',
}


# =============================================================================
# PRÉPARATION DONNÉES
# =============================================================================
MODALITY_TOK = {'spatial': 4}
SPECIE_TOK   = {'mouse': 6}
ASSAY_TOK    = {'cosmx': 8, 'merfish': 7}


def prepare_unified(input_path, output_path, model_path, raw_label_col,
                    mapping_to_unified, modality, specie, assay,
                    is_train=True, seed=42):
    print(f'\n[prep] {input_path.name}')
    adata = ad.read_h5ad(str(input_path))
    print(f'   {adata.n_obs:,} cellules x {adata.n_vars:,} gènes')

    raw_labels = adata.obs[raw_label_col].astype(str).values
    unified = pd.Series(raw_labels).map(mapping_to_unified)
    n_unmapped = unified.isna().sum()
    if n_unmapped > 0:
        keep = unified.notna().values
        adata = adata[keep].copy()
        unified = unified[keep]

    print(f'   distribution finale :')
    for cls, count in unified.value_counts().items():
        print(f'      {cls:18s} : {count:,}')

    model_h5ad = ad.read_h5ad(str(model_path))
    #merged = ad.concat([model_h5ad, adata], join='outer', axis=0)[1:].copy()
    n_model = model_h5ad.n_obs
    merged = ad.concat([model_h5ad, adata], join='outer', axis=0)[n_model:].copy()
    merged = merged[:, model_h5ad.var_names].copy()

    merged.obs['modality'] = MODALITY_TOK[modality]
    merged.obs['specie']   = SPECIE_TOK[specie]
    merged.obs['assay']    = ASSAY_TOK[assay]

    n = merged.n_obs
    if is_train:
        rng = np.random.default_rng(seed)
        idx = np.arange(n); rng.shuffle(idx)
        n_tr, n_va = int(0.7 * n), int(0.15 * n)
        splits = np.empty(n, dtype=object)
        splits[idx[:n_tr]] = 'train'
        splits[idx[n_tr:n_tr+n_va]] = 'val'
        splits[idx[n_tr+n_va:]] = 'test'
    else:
        splits = ['test'] * n
    merged.obs['nicheformer_split'] = pd.Categorical(
        splits, categories=['train', 'val', 'test'])

    merged.obs['cell_type_unified'] = unified.values
    merged.obs['label'] = np.array(
        [UNIFIED_CLASSES[lab] for lab in unified.values], dtype=np.int64,
    )
    merged.write_h5ad(str(output_path))
    print(f'   sauvegardé : {output_path.name} ({merged.n_obs:,} cellules)')


# Vérifier si déjà préparé
if not (PROCESSED_DIR / 'cosmx_prepared.h5ad').exists():
    prepare_unified(
        input_path=RAW_DIR / 'cosmx_mouse_brain.h5ad',
        output_path=PROCESSED_DIR / 'cosmx_prepared.h5ad',
        model_path=PRETRAINED_DIR / 'model.h5ad',
        raw_label_col='cell_type', mapping_to_unified=COSMX_TO_UNIFIED,
        modality='spatial', specie='mouse', assay='cosmx', is_train=True,
    )
else:
    print('cosmx_prepared.h5ad déjà existant -> skip prep CosMx')

if not (PROCESSED_DIR / 'merfish_prepared.h5ad').exists():
    prepare_unified(
        input_path=RAW_DIR / 'merfish_mouse_brain.h5ad',
        output_path=PROCESSED_DIR / 'merfish_prepared.h5ad',
        model_path=PRETRAINED_DIR / 'model.h5ad',
        raw_label_col='Cell_class', mapping_to_unified=MERFISH_TO_UNIFIED,
        modality='spatial', specie='mouse', assay='merfish', is_train=False,
    )
else:
    print('merfish_prepared.h5ad déjà existant -> skip prep MERFISH')

label_to_idx = UNIFIED_CLASSES
with open(PROCESSED_DIR / 'label_mapping.json', 'w') as f:
    json.dump(label_to_idx, f, indent=2)
n_classes = len(label_to_idx)


# =============================================================================
# FIX 1 : macro_f1 corrigé (compte F1=0 pour classes mal prédites)
# =============================================================================
def macro_f1(preds, labels, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = float(((preds == c) & (labels == c)).sum())
        fp = float(((preds == c) & (labels != c)).sum())
        fn = float(((preds != c) & (labels == c)).sum())
        # Skip uniquement si la classe est totalement absente du dataset
        if tp + fp == 0 and tp + fn == 0:
            continue
        # Sinon, si modèle ne prédit jamais cette classe OU rate toutes les
        # vraies instances → F1 = 0 (ne pas SKIP comme avant !)
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        p, r = tp / (tp + fp), tp / (tp + fn)
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


# =============================================================================
# MODULES LoRA (inchangés)
# =============================================================================
class LoRALinear(nn.Module):
    def _init_(self, base, r=8, alpha=16, dropout=0.0):
        super()._init_()
        self.base = base
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.scaling * (
            self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t())


class LoRAMultiheadAttention(nn.Module):
    def _init_(self, base_mha, r=8, alpha=16, dropout=0.0, apply_to_out=True):
        super()._init_()
        self.base = base_mha
        self.embed_dim = base_mha.embed_dim
        self.num_heads = base_mha.num_heads
        self.batch_first = base_mha.batch_first
        self._qkv_same_embed_dim = getattr(base_mha, '_qkv_same_embed_dim', True)
        self.in_proj_weight = base_mha.in_proj_weight
        self.in_proj_bias   = base_mha.in_proj_bias
        self.out_proj       = base_mha.out_proj
        self.bias_k         = base_mha.bias_k
        self.bias_v         = base_mha.bias_v
        self.add_zero_attn  = base_mha.add_zero_attn
        self.dropout        = base_mha.dropout
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        self.apply_to_out = apply_to_out
        E = self.embed_dim
        self.lora_A_q = nn.Parameter(torch.zeros(r, E))
        self.lora_B_q = nn.Parameter(torch.zeros(E, r))
        self.lora_A_k = nn.Parameter(torch.zeros(r, E))
        self.lora_B_k = nn.Parameter(torch.zeros(E, r))
        self.lora_A_v = nn.Parameter(torch.zeros(r, E))
        self.lora_B_v = nn.Parameter(torch.zeros(E, r))
        for p in [self.lora_A_q, self.lora_A_k, self.lora_A_v]:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        if apply_to_out:
            self.out_lora = LoRALinear(base_mha.out_proj, r=r, alpha=alpha, dropout=dropout)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False

    def _delta_qkv(self, x):
        x_d = self.lora_dropout(x)
        dq = x_d @ self.lora_A_q.t() @ self.lora_B_q.t()
        dk = x_d @ self.lora_A_k.t() @ self.lora_B_k.t()
        dv = x_d @ self.lora_A_v.t() @ self.lora_B_v.t()
        return self.scaling * torch.cat([dq, dk, dv], dim=-1)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None, is_causal=False):
        E = self.embed_dim
        qkv = F.linear(query, self.base.in_proj_weight, self.base.in_proj_bias)
        qkv = qkv + self._delta_qkv(query)
        q, k, v = qkv.chunk(3, dim=-1)
        if not self.batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        B, L, _ = q.shape
        head_dim = E // self.num_heads
        q = q.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        attn_mask_add = None
        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                attn_mask_add = torch.zeros(B, 1, 1, L, dtype=q.dtype, device=q.device)
                attn_mask_add.masked_fill_(
                    key_padding_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
            else:
                attn_mask_add = key_padding_mask.to(q.dtype).unsqueeze(1).unsqueeze(1)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask_add,
            dropout_p=self.base.dropout if self.training else 0.0,
            is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, E)
        if not self.batch_first:
            attn_out = attn_out.transpose(0, 1)
        if self.apply_to_out:
            attn_out = self.out_lora(attn_out)
        else:
            attn_out = self.base.out_proj(attn_out)
        return attn_out, None


def inject_lora(model, r=8, alpha=16, dropout=0.0):
    for layer in model.encoder.layers:
        layer.self_attn = LoRAMultiheadAttention(
            layer.self_attn, r=r, alpha=alpha, dropout=dropout)
    print(f'[LoRA] {len(model.encoder.layers)} couches adaptées (r={r}, alpha={alpha})')


def freeze_except_lora_and_head(model, head_names=('head',)):
    n_train = n_total = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        is_lora = 'lora_' in name
        is_head = any(h in name.split('.') for h in head_names)
        p.requires_grad = is_lora or is_head
        if p.requires_grad:
            n_train += p.numel()
    return n_train, n_total


# =============================================================================
# FIX 2 : Class weights pour CrossEntropyLoss
# =============================================================================
def compute_class_weights(adata_path, n_classes, device='cpu'):
    """Calcule les weights pour CrossEntropyLoss à partir des fréquences."""
    a = ad.read_h5ad(str(adata_path))
    if 'nicheformer_split' in a.obs.columns:
        train_labels = a.obs.loc[a.obs['nicheformer_split'] == 'train', 'label'].values
    else:
        train_labels = a.obs['label'].values
    counts = np.bincount(train_labels.astype(int), minlength=n_classes).astype(np.float32)
    weights = 1.0 / counts
    # Normalisation : la moyenne des weights vaut 1 (pas plus de gradient global)
    weights = weights / weights.mean()
    print(f'\nClass weights (1/freq, normalisé) :')
    for i, w in enumerate(weights):
        cls = list(UNIFIED_CLASSES.keys())[i]
        print(f'  {cls:18s} : count={int(counts[i]):6d}  weight={w:.3f}')
    return torch.tensor(weights, dtype=torch.float32, device=device)


# =============================================================================
# LIGHTNING MODULE (avec class weights)
# =============================================================================
class NicheformerLoRA(pl.LightningModule):
    def _init_(self, backbone, n_classes, class_weights=None,
                 lora_r=8, lora_alpha=16, lora_dropout=0.05,
                 head_dropout=0.1, lr=5e-4, weight_decay=0.01,
                 warmup_steps=200, max_steps=5000, without_context=True):
        super()._init_()
        self.save_hyperparameters(ignore=['backbone', 'class_weights'])
        self.backbone = backbone
        self.backbone.hparams.masking_p = 0.0
        inject_lora(self.backbone, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        d = self.backbone.hparams.dim_model
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.Tanh(), nn.Dropout(head_dropout),
            nn.Linear(d, n_classes),
        )
        n_train, n_total = freeze_except_lora_and_head(self, head_names=('head',))
        print(f'[FT-LoRA] entraînables : {n_train:,} / {n_total:,} '
              f'({100*n_train/n_total:.3f}%)')

        # FIX 2 : CE avec class_weights pour combattre le déséquilibre
        if class_weights is not None:
            self.register_buffer('_class_weights', class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            print(f'[CE] CrossEntropyLoss avec class_weights activé')
        else:
            self.criterion = nn.CrossEntropyLoss()
        self._buf = {'val_logits': [], 'val_labels': [],
                     'test_logits': [], 'test_labels': []}

    def _encode(self, batch):
        batch = self.backbone.on_after_batch_transfer(batch, 0)
        batch = complete_masking(batch, p=0.0,
                                  n_tokens=self.backbone.hparams.n_tokens + 5)
        x = self.backbone.embeddings(batch['masked_indices'])
        if self.backbone.hparams.learnable_pe:
            pos = self.backbone.pos.to(x.device)
            x = self.backbone.dropout(x + self.backbone.positional_embedding(pos))
        else:
            x = self.backbone.positional_embedding(x)
        for layer in self.backbone.encoder.layers:
            x = layer(x, src_key_padding_mask=batch['attention_mask'],
                      is_causal=False)
        if self.hparams.without_context:
            x = x[:, 3:, :]
        return x.mean(dim=1)

    def forward(self, batch):
        return self.head(self._encode(batch))

    def _step(self, batch):
        labels = batch['label'].long()
        logits = self.forward(batch)
        return self.criterion(logits, labels), logits, labels

    def training_step(self, batch, _):
        loss, logits, labels = self._step(batch)
        with torch.no_grad():
            acc = (logits.argmax(-1) == labels).float().mean()
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/acc', acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, logits, labels = self._step(batch)
        self.log('val/loss', loss, prog_bar=True)
        self._buf['val_logits'].append(logits.detach().cpu())
        self._buf['val_labels'].append(labels.detach().cpu())

    def on_validation_epoch_end(self):
        if not self._buf['val_logits']:
            return
        logits = torch.cat(self._buf['val_logits'])
        labels = torch.cat(self._buf['val_labels'])
        preds = logits.argmax(-1)
        acc = (preds == labels).float().mean()
        f1 = macro_f1(preds.numpy(), labels.numpy(), self.hparams.n_classes)
        # On log aussi le nombre de classes prédites (pour détecter dégénérescence)
        n_classes_pred = len(np.unique(preds.numpy()))
        self.log('val/acc', acc, prog_bar=True)
        self.log('val/f1_macro', f1, prog_bar=True)
        self.log('val/n_classes_pred', float(n_classes_pred), prog_bar=True)
        self.log('val_f1_macro', f1)
        self._buf['val_logits'].clear()
        self._buf['val_labels'].clear()

    def test_step(self, batch, _):
        loss, logits, labels = self._step(batch)
        self.log('test/loss', loss)
        self._buf['test_logits'].append(logits.detach().cpu())
        self._buf['test_labels'].append(labels.detach().cpu())

    def on_test_epoch_end(self):
        if not self._buf['test_logits']:
            return
        logits = torch.cat(self._buf['test_logits'])
        labels = torch.cat(self._buf['test_labels'])
        preds = logits.argmax(-1)
        acc = (preds == labels).float().mean()
        f1 = macro_f1(preds.numpy(), labels.numpy(), self.hparams.n_classes)
        self.log('test/acc', acc)
        self.log('test/f1_macro', f1)
        self.last_preds = preds.numpy()
        self.last_labels = labels.numpy()
        self.last_logits = logits.numpy()
        self._buf['test_logits'].clear()
        self._buf['test_labels'].clear()

    def configure_optimizers(self):
        head_params = list(self.head.parameters())
        lora_params = [p for name, p in self.named_parameters() if 'lora_' in name]
    
        opt = optim.AdamW([
            {'params': lora_params, 'lr': self.hparams.lr}, 
            {'params': head_params, 'lr': self.hparams.lr * 10} # 5e-3 au lieu de 5e-4
        ], weight_decay=self.hparams.weight_decay)

        warmup = self.hparams.warmup_steps
        total = max(self.hparams.max_steps, warmup + 1)
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            prog = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * prog))
        sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}


# =============================================================================
# CONFIG (FIX 3 : lr 5e-4, accumulate 2)
# =============================================================================
CFG = dict(
    max_seq_len=1500,
    batch_size=16,
    num_workers=2,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lr=5e-4,                        # ← FIX 3 : ÉTAIT 1e-4
    weight_decay=0.01,
    warmup_steps=50,               # ← un peu plus
    max_epochs=15,
    early_stop_patience=5,
    head_dropout=0.1,
    accumulate_grad_batches=2,      # ← FIX 3 : ÉTAIT 4
    precision='16-mixed',
)

RUN_DIR = OUTPUT_DIR / 'run_fix'
RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RUN_DIR / 'checkpoints'
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATALOADERS
# =============================================================================
adata = ad.read_h5ad(str(PROCESSED_DIR / 'cosmx_prepared.h5ad'))
tech_mean = np.load(str(PRETRAINED_DIR / 'cosmx_mean_script.npy'))

ds_kwargs = dict(
    adata=adata, technology_mean=tech_mean,
    max_seq_len=CFG['max_seq_len'], aux_tokens=30, chunk_size=1000,
    metadata_fields={'obs': ['modality', 'specie', 'assay', 'label']},
)
train_ds = NicheformerDataset(split='train', **ds_kwargs)
val_ds = NicheformerDataset(split='val', **ds_kwargs)
test_ds = NicheformerDataset(split='test', **ds_kwargs)
print(f'\ntrain={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}')

train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True,
                          num_workers=CFG['num_workers'], pin_memory=True,
                          drop_last=True)
val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False,
                         num_workers=CFG['num_workers'], pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=CFG['batch_size'], shuffle=False,
                          num_workers=CFG['num_workers'], pin_memory=True)


# =============================================================================
# MODÈLE + ENTRAÎNEMENT
# =============================================================================
pl.seed_everything(42, workers=True)

# FIX 2 : calcul des class weights
class_weights = compute_class_weights(
    PROCESSED_DIR / 'cosmx_prepared.h5ad', n_classes,
)

backbone = Nicheformer.load_from_checkpoint(
    str(PRETRAINED_DIR / 'nicheformer.ckpt'),
    strict=False, weights_only=False,
)

max_steps = (len(train_loader) // CFG['accumulate_grad_batches']) * CFG['max_epochs']
model = NicheformerLoRA(
    backbone=backbone, n_classes=n_classes,
    class_weights=class_weights,    # ← FIX 2
    lora_r=CFG['lora_r'], lora_alpha=CFG['lora_alpha'],
    lora_dropout=CFG['lora_dropout'], head_dropout=CFG['head_dropout'],
    lr=CFG['lr'], weight_decay=CFG['weight_decay'],
    warmup_steps=CFG['warmup_steps'], max_steps=max_steps,
)

checkpoint_callback = ModelCheckpoint(
    dirpath=str(CKPT_DIR),
    filename='best-{epoch:02d}-{val_f1_macro:.3f}',
    monitor='val/f1_macro', mode='max',
    save_top_k=1, save_last=True,
    auto_insert_metric_name=False,
)

trainer = pl.Trainer(
    max_epochs=CFG['max_epochs'],
    accelerator='gpu', devices=1,
    precision=CFG['precision'],
    gradient_clip_val=1.0,
    accumulate_grad_batches=CFG['accumulate_grad_batches'],
    callbacks=[
        checkpoint_callback,
        EarlyStopping(monitor='val/f1_macro', mode='max',
                      patience=CFG['early_stop_patience']),
        LearningRateMonitor(logging_interval='step'),
    ],
    logger=CSVLogger(save_dir=str(RUN_DIR), name='logs'),
    log_every_n_steps=10,
)

print('\n>>> ENTRAÎNEMENT')
trainer.fit(model, train_loader, val_loader)

best_ckpt_path = checkpoint_callback.best_model_path
print(f'\n>>> Meilleur checkpoint : {best_ckpt_path}')
print(f'>>> Score : {checkpoint_callback.best_model_score}')


# =============================================================================
# ÉVALUATION
# =============================================================================
print('\n>>> ÉVAL IN-DOMAIN')
trainer.test(model, test_loader, ckpt_path=best_ckpt_path)
np.save(RUN_DIR / 'preds_indomain.npy', model.last_preds)
np.save(RUN_DIR / 'labels_indomain.npy', model.last_labels)

# Distribution des prédictions
unique_id, counts_id = np.unique(model.last_preds, return_counts=True)
idx_to_label = {v: k for k, v in label_to_idx.items()}
print('Distribution of in-domain predictions:')
for u, c in zip(unique_id, counts_id):
    print(f'  {idx_to_label[int(u)]:18s} : {int(c):,}')

print('\n>>> ÉVAL CROSS-TECH MERFISH')
adata_m = ad.read_h5ad(str(PROCESSED_DIR / 'merfish_prepared.h5ad'))
tech_m = np.load(str(PRETRAINED_DIR / 'merfish_mean_script.npy'))

merfish_ds = NicheformerDataset(
    adata=adata_m, technology_mean=tech_m, split='test',
    max_seq_len=CFG['max_seq_len'], aux_tokens=30, chunk_size=1000,
    metadata_fields={'obs': ['modality', 'specie', 'assay', 'label']},
)
merfish_loader = DataLoader(merfish_ds, batch_size=CFG['batch_size'],
                             shuffle=False, num_workers=CFG['num_workers'],
                             pin_memory=True)

trainer.test(model, merfish_loader, ckpt_path=best_ckpt_path)
np.save(RUN_DIR / 'preds_crosstech.npy', model.last_preds)
np.save(RUN_DIR / 'labels_crosstech.npy', model.last_labels)

unique_ct, counts_ct = np.unique(model.last_preds, return_counts=True)
print('Distribution of cross-tech predictions:')
for u, c in zip(unique_ct, counts_ct):
    print(f'  {idx_to_label[int(u)]:18s} : {int(c):,}')


# =============================================================================
# FIGURES
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

FIG_DIR = RUN_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# Courbes
metrics_csv = list((RUN_DIR / 'logs').rglob('metrics.csv'))[0]
df = pd.read_csv(metrics_csv)
agg = df.groupby('epoch').mean(numeric_only=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
if 'train/loss_epoch' in agg:
    axes[0].plot(agg.index, agg['train/loss_epoch'], 'o-', label='train')
if 'val/loss' in agg:
    axes[0].plot(agg.index, agg['val/loss'], 's-', label='val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_title('Loss')

if 'val/acc' in agg:
    axes[1].plot(agg.index, agg['val/acc'], 's-', label='val acc')
if 'val/f1_macro' in agg:
    axes[1].plot(agg.index, agg['val/f1_macro'], '^-', label='val F1 macro', linewidth=2)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Score')
axes[1].set_ylim(0, 1); axes[1].legend()
axes[1].grid(alpha=0.3); axes[1].set_title('Accuracy & F1')
fig.suptitle("Training curves — Nicheformer + LoRA + class weights")
fig.tight_layout()
fig.savefig(FIG_DIR / 'train_curves.png', dpi=150, bbox_inches='tight')

# Reste des figures
preds_id = np.load(RUN_DIR / 'preds_indomain.npy')
labels_id = np.load(RUN_DIR / 'labels_indomain.npy')
preds_ct = np.load(RUN_DIR / 'preds_crosstech.npy')
labels_ct = np.load(RUN_DIR / 'labels_crosstech.npy')

acc_id = (preds_id == labels_id).mean()
f1_id = macro_f1(preds_id, labels_id, n_classes)
acc_ct = (preds_ct == labels_ct).mean()
f1_ct = macro_f1(preds_ct, labels_ct, n_classes)

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(2); w = 0.35
ax.bar(x - w/2, [acc_id, acc_ct], w, label='Accuracy', color='#1f77b4')
ax.bar(x + w/2, [f1_id, f1_ct], w, label='F1 macro', color='#ff7f0e')
ax.set_xticks(x)
ax.set_xticklabels(['In-domain\n(CosMx)', 'Cross-tech\n(MERFISH)'])
ax.set_ylabel('Score'); ax.set_ylim(0, 1); ax.legend()
ax.set_title('Performance in-domain vs cross-tech')
for i, v in enumerate([acc_id, acc_ct]):
    ax.text(i - w/2, v + 0.01, f'{v:.3f}', ha='center')
for i, v in enumerate([f1_id, f1_ct]):
    ax.text(i + w/2, v + 0.01, f'{v:.3f}', ha='center')
fig.tight_layout()
fig.savefig(FIG_DIR / 'comparison.png', dpi=150, bbox_inches='tight')

class_names = [idx_to_label[i] for i in range(n_classes)]
for name, preds, labels in [('indomain', preds_id, labels_id),
                              ('crosstech', preds_ct, labels_ct)]:
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)),
                          normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, annot=True, fmt='.2f',
                annot_kws={'size': 9}, cbar=True, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion matrix — {name}')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'confusion_{name}.png', dpi=150, bbox_inches='tight')

summary = f"""
==================================================================
RÉSULTATS — Nicheformer + LoRA (r={CFG['lora_r']}, alpha={CFG['lora_alpha']})
                  AVEC class weights + lr=5e-4
==================================================================

In-domain  (CosMx test)    : accuracy = {acc_id:.4f}, F1 macro = {f1_id:.4f}
Cross-tech (MERFISH all)   : accuracy = {acc_ct:.4f}, F1 macro = {f1_ct:.4f}
Drop F1 cross-tech         : {(f1_id - f1_ct)*100:.1f} points

Distribution des prédictions in-domain :
"""
for u, c in zip(unique_id, counts_id):
    summary += f'  {idx_to_label[int(u)]:18s} : {int(c):,}\n'
summary += "\nDistribution des prédictions cross-tech :\n"
for u, c in zip(unique_ct, counts_ct):
    summary += f'  {idx_to_label[int(u)]:18s} : {int(c):,}\n'
summary += '==================================================================\n'

print(summary)
with open(RUN_DIR / 'summary.txt', 'w') as f:
    f.write(summary)
print(f'\n>>> Tout est dans : {RUN_DIR}')