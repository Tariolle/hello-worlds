# eeg_jepa_tuab benchmark

Primary metric: `balanced_acc`. Higher is better.

Local protocol: Full TUAB_PREPROCESSED patient-disjoint train/eval split, recording-level frozen probe.

Caution: Do not mix full-split balanced accuracy with subset accuracy without marking the protocol.

| Rank | Method | Status | Protocol | Acc | BalAcc | AUROC | F1 | Comparable? | Source |
|---:|---|---|---|---:|---:|---:|---:|---|---|
| 1 | LuMamba | published_reference | TUAB abnormality detection, reported balanced accuracy; protocol differs from this repo until reproduced. |  | 0.8099 |  |  | reference only | arXiv 2603.19100 abstract |
| 2 | Laya-S | published_reference | EEG-Bench frozen linear probe, Abnormal task; not the full local TUAB split. |  | 0.7980 |  |  | reference only | references/laya/SUMMARY.md |
|  | JEPA VICReg ambient | pending_training | Full TUAB_PREPROCESSED patient-disjoint train/eval split, recording-level frozen probe. |  |  |  |  | yes | local |
|  | JEPA SIGReg ambient | pending_training | Full TUAB_PREPROCESSED patient-disjoint train/eval split, recording-level frozen probe. |  |  |  |  | yes | local |
|  | JEPA SIGReg tangent | pending_training | Full TUAB_PREPROCESSED patient-disjoint train/eval split, recording-level frozen probe. |  |  |  |  | yes | local |
|  | JEPA PEIRA ambient | pending_training | Full TUAB_PREPROCESSED patient-disjoint train/eval split, recording-level frozen probe. |  |  |  |  | yes | local |
|  | JEPA PEIRA tangent | pending_training | Full TUAB_PREPROCESSED patient-disjoint train/eval split, recording-level frozen probe. |  |  |  |  | yes | local |
|  | Riemannian covariance + logistic probe | runnable_local | Full TUAB_PREPROCESSED patient-disjoint train/eval split, recording-level frozen probe. |  |  |  |  | yes | local |
|  | Random EEGEncoder1D floor | runnable_local | Full TUAB_PREPROCESSED patient-disjoint train/eval split, recording-level frozen probe. |  |  |  |  | yes | local |
|  | EEG-VJEPA frozen | published_reference | TUAB subset, attention-pooling frozen probe; train 276 normal/270 abnormal, val 150 normal/126 abnormal. | 0.8330 |  | 0.8770 | 0.8240 | reference only | references/eeg-vjepa/SUMMARY.md |
|  | EEG-VJEPA fine-tuned | published_reference | TUAB subset, full fine-tuning; train 276 normal/270 abnormal, val 150 normal/126 abnormal. | 0.8580 |  | 0.8850 | 0.8560 | reference only | references/eeg-vjepa/SUMMARY.md |
|  | LaBraM fine-tuned | published_reference | Reported in EEG-VJEPA TUAB subset table. | 0.8258 |  | 0.9204 |  | reference only | references/eeg-vjepa/SUMMARY.md |
|  | ChronoNet supervised | published_reference | Reported in EEG-VJEPA TUAB subset table. | 0.8657 |  |  |  | reference only | references/eeg-vjepa/SUMMARY.md |
|  | BioSerenity-E1 fine-tuned | published_reference | Fine-tuned on private data, validated on TUAB eval dataset; TUAB train was not used. | 0.8225 |  |  |  | reference only | arXiv 2505.21507 abstract |

## Notes

- **LuMamba**: Efficient topology-invariant Mamba/LeJEPA-style reference; use as a target, not a local apples-to-apples row.
- **Laya-S**: Closest conceptual reference: LeJEPA/SIGReg EEG latent prediction.
- **JEPA VICReg ambient**: Reference arm using eb_jepa's default VICReg-style anti-collapse loss.
- **JEPA SIGReg ambient**: Laya-like local baseline: SIGReg in Euclidean pooled-representation space.
- **JEPA SIGReg tangent**: Geometry-aware SIGReg on SPD log-Euclidean tangent features.
- **JEPA PEIRA ambient**: Factorial control: PEIRA regularization without SPD tangent geometry.
- **JEPA PEIRA tangent**: Hypothesis cell: distribution-free anti-collapse on SPD tangent features.
- **Riemannian covariance + logistic probe**: 0-parameter complexity yardstick; run before GPU experiments to sanity-check EDF loading.
- **Random EEGEncoder1D floor**: Same architecture as the JEPA encoder, untrained; confirms the probe is not leaking labels.
- **EEG-VJEPA frozen**: Strong JEPA TUAB reference, but uses a much smaller subset and reports accuracy.
- **EEG-VJEPA fine-tuned**: Not directly comparable to a frozen probe from this repo.
- **LaBraM fine-tuned**: Hard AUROC target; protocol and adaptation differ.
- **ChronoNet supervised**: Supervised upper-reference for subset accuracy, not a foundation-model frozen probe.
- **BioSerenity-E1 fine-tuned**: Useful clinical deployment reference, but not a local TUAB-train/TUAB-eval comparison.
