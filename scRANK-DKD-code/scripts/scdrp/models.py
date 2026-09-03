"""Lightweight program-aware DeepSets model with attention pooling.

Deliberately not a Transformer or a GNN: the available number of patients is small, so
the parameter count has to stay far below a conventional deep model, or external
validation is unreliable and any improvement cannot be attributed. Kept under 100k
parameters."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DeepSetsConfig:
    n_genes: int
    n_cells: int
    n_programs: int
    n_categories: int = 3
    gene_dim: int = 4
    label_dim: int = 4
    hidden: int = 16
    dropout: float = 0.3
    use_program: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-3
    epochs: int = 300
    patience: int = 40
    batch_size: int = 16
    entropy_weight: float = 1e-3
    pos_weight: float = 1.0


class ProgramAwareDeepSets(nn.Module):
    def __init__(self, cfg: DeepSetsConfig):
        super().__init__()
        self.cfg = cfg
        self.gene_emb = nn.Embedding(cfg.n_genes, cfg.gene_dim)
        if cfg.use_program:
            self.cell_emb = nn.Embedding(cfg.n_cells, cfg.label_dim)
            self.prog_emb = nn.Embedding(cfg.n_programs, cfg.label_dim)
            self.cat_emb = nn.Embedding(cfg.n_categories, cfg.label_dim)
            extra = 5 * cfg.label_dim
        else:
            extra = 0
        in_dim = 2 + 2 * cfg.gene_dim + extra

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden), nn.LayerNorm(cfg.hidden),
            nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden))
        self.att_w = nn.Linear(cfg.hidden, cfg.hidden)
        self.att_v = nn.Linear(cfg.hidden, 1, bias=False)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout),
                                  nn.Linear(cfg.hidden, 1))

    def tokens(self, P, M, lab):
        n = P.shape[0]
        feats = [P.unsqueeze(-1), M.unsqueeze(-1),
                 self.gene_emb(lab["ga"]).expand(n, -1, -1),
                 self.gene_emb(lab["gb"]).expand(n, -1, -1)]
        if self.cfg.use_program:
            feats += [self.cell_emb(lab["ca"]).expand(n, -1, -1),
                      self.cell_emb(lab["cb"]).expand(n, -1, -1),
                      self.prog_emb(lab["pa"]).expand(n, -1, -1),
                      self.prog_emb(lab["pb"]).expand(n, -1, -1),
                      self.cat_emb(lab["cat"]).expand(n, -1, -1)]
        return torch.cat(feats, dim=-1)

    def forward(self, P, M, mask, lab, return_attention=False):
        h = self.encoder(self.tokens(P, M, lab))
        a = self.att_v(torch.tanh(self.att_w(h))).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(a, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        z = (alpha.unsqueeze(-1) * h).sum(1)
        logit = self.head(z).squeeze(-1)
        if return_attention:
            return logit, alpha, h
        return logit, alpha


def _entropy(alpha: torch.Tensor) -> torch.Tensor:
    a = alpha.clamp_min(1e-9)
    return -(a * a.log()).sum(1).mean()


def train_deepsets(cfg: DeepSetsConfig, train, valid, labels, device="cpu",
                   seed: int = 20260722, fixed_epochs: int | None = None
                   ) -> tuple[ProgramAwareDeepSets, int]:
    """Train the ensemble and return out-of-fold logits alongside the held-out predictions.
    
    A single model trained on 37 to 83 biopsies inverted its sign on some held-out
    cohorts, so predictions come from an ensemble; both sides average the same number of
    equally sized models, which keeps the logit scales comparable."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ProgramAwareDeepSets(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    lossfn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(cfg.pos_weight, dtype=torch.float32))
    n = train["P"].shape[0]

    from sklearn.metrics import roc_auc_score
    y_val = valid["y"].numpy() if valid is not None else None
    best_state, best_score, wait, best_epoch = None, -np.inf, 0, cfg.epochs
    total_epochs = fixed_epochs if fixed_epochs else cfg.epochs
    for epoch in range(total_epochs):
        model.train()
        perm = torch.randperm(n)
        for k in range(0, n, cfg.batch_size):
            idx = perm[k:k + cfg.batch_size]
            logit, alpha = model(train["P"][idx], train["M"][idx],
                                 train["mask"][idx], labels)
            loss = lossfn(logit, train["y"][idx])
            loss = loss - cfg.entropy_weight * _entropy(alpha)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        if valid is None:
            continue
        model.eval()
        with torch.no_grad():
            logit, _ = model(valid["P"], valid["M"], valid["mask"], labels)
            if len(np.unique(y_val)) > 1:
                score = float(roc_auc_score(y_val, logit.numpy()))
            else:
                score = -float(lossfn(logit, valid["y"]))
        if score > best_score + 1e-5:
            best_score, wait, best_epoch = score, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (fixed_epochs or best_epoch)


@torch.no_grad()
def predict(model: ProgramAwareDeepSets, batch, labels, device="cpu"):
    """Predicted probabilities for one batch of pair tokens."""
    model.eval()
    logit, alpha = model(batch["P"], batch["M"], batch["mask"], labels)
    lg = logit.cpu().numpy()
    return lg, 1.0 / (1.0 + np.exp(-lg)), alpha.cpu().numpy()


def fit_platt(logit: np.ndarray, y: np.ndarray):
    """Fit a Platt scaling on out-of-fold logits."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y)
    return lambda z: lr.predict_proba(np.asarray(z).reshape(-1, 1))[:, 1]


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
