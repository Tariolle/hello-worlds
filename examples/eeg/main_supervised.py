"""EEG — SUPERVISED-from-scratch baseline (the honest yardstick for SSL).

Trains the SAME ``EEGEncoder1D`` end-to-end with a linear classification head and
cross-entropy on TUAB-train labels, then reports TWO numbers on the patient-disjoint
TUAB-eval split:

  * END-TO-END         — encoder+head trained jointly, recording-level prediction
                         (mean-pooled window logits). The "normal supervised" topline.
  * FROZEN-PROBE        — run ``examples.eeg.eval`` on this checkpoint afterwards: it
                         freezes the supervised encoder and fits the SAME linear probe
                         we use for SSL. Apples-to-apples with our SSL 0.819 — isolates
                         which OBJECTIVE (supervised CE vs SSL) makes better frozen
                         features. The checkpoint is saved in the SSL format so eval.py
                         reads it unchanged.

FAIRNESS CONTRACT (so "supervised overfits" is a finding, not an artifact):
  * same architecture, same TUAB-train data, same augmentations as the SSL views
    (noise / scale-jitter / chan-drop / time-mask), same patient-disjoint eval;
  * EARLY STOPPING on a patient-disjoint dev split carved from TRAIN patients only
    (15% of patients, fixed dev_seed) — never peeking at eval. This is the leakage
    gate; the dev split uses the basename ``patient`` token like label_efficiency.py;
  * class-balanced cross-entropy (mirrors the probe's class_weight='balanced').

Run (GPU):  python -u -m examples.eeg.main_supervised --fname examples/eeg/cfgs/train_supervised.yaml
            python -u -m examples.eeg.eval --ckpt <ckpt_dir>/latest.pth.tar   # frozen-probe number
"""
import os
import sys
import random as _random

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.eeg.dataset import EEGConfig, EEGDataset
from examples.eeg.main import build_encoder


def _patient(path):
    """Patient id = first underscore token of the basename (matches label_efficiency.py)."""
    return os.path.basename(path).split("_")[0]


def _augment_batch(x, c):
    """Match dataset._augment on a [B, C, T] device tensor (uses the global torch RNG,
    seeded in run()). Same four augmentations the SSL views see, so the supervised
    baseline gets the SAME regularisation — its strongest, fairest shot."""
    B, C, T = x.shape
    if c.aug_scale_jitter > 0:
        x = x * (1.0 + (torch.rand(B, C, 1, device=x.device) * 2 - 1) * c.aug_scale_jitter)
    if c.aug_noise_std > 0:
        x = x + torch.randn(B, C, T, device=x.device) * c.aug_noise_std
    if c.aug_chan_drop_p > 0:
        x = x * (torch.rand(B, C, 1, device=x.device) > c.aug_chan_drop_p).float()
    if c.aug_time_mask_frac > 0:
        mlen = (torch.rand(B, device=x.device) * c.aug_time_mask_frac * T).long()
        start = (torch.rand(B, device=x.device) * (T - mlen).clamp(min=1).float()).long()
        ar = torch.arange(T, device=x.device)[None, :]
        m = (ar >= start[:, None]) & (ar < (start + mlen)[:, None])
        x = x.masked_fill(m[:, None, :], 0.0)
    return x


def _loader(dcfg_container, items, num_workers, batch_size, shuffle):
    """Supervised/probe dataset restricted to an explicit (path,label) item list."""
    cfg = EEGConfig(**dcfg_container)
    cfg.split, cfg.mode, cfg.num_workers, cfg.batch_size = "train", "supervised", num_workers, batch_size
    ds = EEGDataset(cfg)
    ds.items = items                                     # override the patient partition
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=shuffle, persistent_workers=num_workers > 0)


@torch.no_grad()
def _eval_recordings(encoder, head, dcfg_container, split, device, num_workers):
    """Recording-level end-to-end metric: mean-pool window logits -> predict."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    cfg = EEGConfig(**dcfg_container)
    cfg.split, cfg.mode, cfg.num_workers = split, "supervised", num_workers
    ds = EEGDataset(cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False,
                                         num_workers=num_workers, pin_memory=True)
    encoder.eval(); head.eval()
    ys, preds, probs = [], [], []
    for wins, labels, ok in loader:
        ok = ok.bool()
        if ok.sum() == 0:
            continue
        wins, labels = wins[ok], labels[ok]
        B, N = wins.shape[0], wins.shape[1]
        x = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
        logits = head(encoder.represent(x)).reshape(B, N, -1).mean(dim=1)
        p = logits.softmax(dim=-1)
        ys += labels.tolist()
        preds += p.argmax(dim=-1).cpu().tolist()
        probs += p[:, 1].cpu().tolist()
    ba = balanced_accuracy_score(ys, preds)
    try:
        auroc = roc_auc_score(ys, probs)
    except ValueError:
        auroc = float("nan")
    return float(ba), float(auroc), len(ys)


def run(fname, **overrides):
    cfg = OmegaConf.load(fname)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = int(cfg.meta.seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); _random.seed(seed)

    dcfg = OmegaConf.to_container(cfg.data, resolve=True)
    s = cfg.sup
    nw = int(cfg.data.num_workers)

    # --- patient-disjoint dev split from TRAIN (leakage gate; fixed dev_seed) ---
    base = EEGDataset(EEGConfig(**{**dcfg, "split": "train", "mode": "supervised"}))
    items = list(base.items)
    pats = sorted({_patient(p) for p, _ in items})
    rng = np.random.default_rng(int(s.dev_seed)); rng.shuffle(pats)
    dev_pats = set(pats[: max(1, int(s.dev_frac * len(pats)))])
    fit_items = [it for it in items if _patient(it[0]) not in dev_pats]
    dev_items = [it for it in items if _patient(it[0]) in dev_pats]
    assert not ({_patient(p) for p, _ in fit_items} & {_patient(p) for p, _ in dev_items}), \
        "dev split leaks patients into fit"
    print(f"[sup] patients: {len(pats)} | fit-rec {len(fit_items)} dev-rec {len(dev_items)} "
          f"| dev patients {len(dev_pats)}", flush=True)

    # class-balanced CE from FIT labels (mirror the probe's class_weight='balanced')
    yfit = np.array([lab for _, lab in fit_items])
    n_cls = int(yfit.max()) + 1
    counts = np.bincount(yfit, minlength=n_cls).astype(np.float64)
    w = torch.tensor(len(yfit) / (n_cls * np.clip(counts, 1, None)), dtype=torch.float32, device=device)

    encoder = build_encoder(cfg.model).to(device)
    head = nn.Linear(encoder.out_dim, n_cls).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                            lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay))
    ce = nn.CrossEntropyLoss(weight=w)

    fit_loader = _loader(dcfg, fit_items, nw, int(cfg.data.batch_size), shuffle=True)

    ckpt_dir = cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ba, best_state, bad = -1.0, None, 0
    for epoch in range(int(s.max_epochs)):
        encoder.train(); head.train()
        running, nb = 0.0, 0
        for wins, labels, ok in fit_loader:
            ok = ok.bool()
            if ok.sum() == 0:
                continue
            wins, labels = wins[ok], labels[ok]
            B, N = wins.shape[0], wins.shape[1]
            x = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
            x = _augment_batch(x, cfg.data)
            y = labels.to(device).repeat_interleave(N)
            opt.zero_grad(set_to_none=True)
            loss = ce(head(encoder.represent(x)), y)
            loss.backward(); opt.step()
            running += float(loss.item()); nb += 1
        dev_ba, dev_auroc, ndev = _eval_recordings_on_items(encoder, head, dcfg, dev_items, device, nw)
        print(f"[sup] epoch {epoch} ce={running / max(nb, 1):.4f} "
              f"dev_balacc={dev_ba:.4f} dev_auroc={dev_auroc:.4f} (n={ndev})", flush=True)
        if dev_ba > best_ba:
            best_ba, bad = dev_ba, 0
            best_state = {"encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                          "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                          "epoch": epoch, "dev_balacc": dev_ba}
        else:
            bad += 1
            if bad >= int(s.patience):
                print(f"[sup] early stop at epoch {epoch} (best dev_balacc={best_ba:.4f} "
                      f"@ epoch {best_state['epoch']})", flush=True)
                break

    # restore best, save in the SSL checkpoint format so eval.py reads it unchanged
    encoder.load_state_dict({k: v.to(device) for k, v in best_state["encoder"].items()})
    head.load_state_dict({k: v.to(device) for k, v in best_state["head"].items()})
    torch.save({"epoch": best_state["epoch"], "encoder": best_state["encoder"],
                "head": best_state["head"], "dev_balacc": best_ba,
                "cfg": OmegaConf.to_container(cfg, resolve=True)},
               os.path.join(ckpt_dir, "latest.pth.tar"))

    ev_ba, ev_auroc, nev = _eval_recordings(encoder, head, dcfg, "eval", device, nw)
    print(f"[sup] TRAINED end-to-end (held-out patients): "
          f"{{'balanced_acc': {ev_ba:.4f}, 'auroc': {ev_auroc:.4f}, 'n_eval': {nev}, "
          f"'best_dev_balacc': {best_ba:.4f}}}", flush=True)
    print(f"[sup] done -> {ckpt_dir}/latest.pth.tar  "
          f"(run examples.eeg.eval on it for the FROZEN-PROBE number)", flush=True)


@torch.no_grad()
def _eval_recordings_on_items(encoder, head, dcfg_container, items, device, num_workers):
    """End-to-end recording-level metric on an explicit item list (the dev split)."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    loader = _loader(dcfg_container, items, num_workers, 8, shuffle=False)
    encoder.eval(); head.eval()
    ys, preds, probs = [], [], []
    for wins, labels, ok in loader:
        ok = ok.bool()
        if ok.sum() == 0:
            continue
        wins, labels = wins[ok], labels[ok]
        B, N = wins.shape[0], wins.shape[1]
        x = wins.reshape(B * N, *wins.shape[2:]).to(device, non_blocking=True)
        logits = head(encoder.represent(x)).reshape(B, N, -1).mean(dim=1)
        p = logits.softmax(dim=-1)
        ys += labels.tolist()
        preds += p.argmax(dim=-1).cpu().tolist()
        probs += p[:, 1].cpu().tolist()
    ba = balanced_accuracy_score(ys, preds)
    try:
        auroc = roc_auc_score(ys, probs)
    except ValueError:
        auroc = float("nan")
    return float(ba), float(auroc), len(ys)


if __name__ == "__main__":
    argv = sys.argv[1:]
    fname = argv[argv.index("--fname") + 1] if "--fname" in argv else \
        "examples/eeg/cfgs/train_supervised.yaml"
    kv = dict(a.split("=", 1) for a in argv if "=" in a and not a.startswith("--"))
    run(fname, **kv)
