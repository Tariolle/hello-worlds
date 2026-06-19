# Lessons (this project)

- **Check prior art before claiming novelty.** Our original angles were each already
  published: SIGReg-for-EEG (Laya), Riemannian-SSL-for-EEG (EEG-ReMinD). Only the
  *intersection* (geometry-aware, latent-predictive, distribution-free anti-collapse)
  is open. Frame as a controlled comparison, not a "first".
- **Two TUAB numbers exist and are not comparable.** Fine-tuned BalAcc (~0.83 ceiling)
  vs frozen linear-probe (much lower, protocol-dependent). Always report **balanced
  accuracy on the full 2717/276 split** + the probe head. EEG-VJEPA's 83.3% is plain
  accuracy on a reduced subset — not comparable.
- **Don't chase asymptotic accuracy on TUAB** — it's label-noise-capped ~0.85–0.87.
  Compete on complexity (vs the 0-param Riemannian 0.86) and sample/data efficiency.
- **PEIRA fits a manifold tangent better than SIGReg** because it assumes no
  distribution (SIGReg forces isotropic Gaussian, which fights non-Gaussian covariance
  tangents). This is the load-bearing scientific reason for the headline. But PEIRA is
  *competitive, not SOTA* — don't sell it as an accuracy win.
- **Verify cited methods from the source before betting on them.** PEIRA's algorithm and
  arXiv id were confirmed from the paper before implementing; the acronym in our notes
  was wrong.
