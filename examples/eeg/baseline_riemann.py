"""Backward-compatible entry point for the classical Riemannian EEG baseline.

The implementation lives in ``examples.eeg.riemannian``. This wrapper preserves
the old module name used by README, benchmark.py, cluster jobs, and TUEV probes,
while exposing the fuller Riemannian EEG approach:

  * channel covariance per EEG window,
  * Euclidean/log-Euclidean/AIRM recording-level covariance means,
  * MDM or tangent-space logistic regression classification.

Run:
  python -m examples.eeg.baseline_riemann --data-root <TUAB_PREPROCESSED>
  python -m examples.eeg.baseline_riemann --data-root <ROOT> \
      --label-scheme folders --classes normal,seizure,dementia
  python -m examples.eeg.baseline_riemann --data-root <ROOT> --classifier mdm
"""
from __future__ import annotations

import sys

from examples.eeg.riemannian import (
    build_argparser,
    fit_score_riemannian,
    main as _riemannian_main,
    parse_classes as _parse_classes,
    recording_covariances as _recording_covariances_impl,
    run_recording_riemannian,
)


def _cov_data_cfg(
    data_root=None,
    label_scheme="tuab",
    class_names=None,
    n_windows=None,
    n_channels=None,
    sfreq=None,
    window_sec=None,
):
    """Compatibility shim for older notebooks importing this helper."""
    from examples.eeg.riemannian import eeg_config

    return eeg_config(
        data_root=data_root,
        label_scheme=label_scheme,
        class_names=class_names,
        n_windows=n_windows,
        n_channels=n_channels,
        sfreq=sfreq,
        window_sec=window_sec,
    )


def _recording_covariances(
    split,
    data_root=None,
    label_scheme="tuab",
    class_names=None,
    n_windows=None,
    estimator="oas",
    n_channels=None,
    sfreq=None,
    window_sec=None,
    return_label_names=False,
    aggregation="riemann",
):
    return _recording_covariances_impl(
        split=split,
        data_root=data_root,
        label_scheme=label_scheme,
        class_names=class_names,
        n_windows=n_windows,
        estimator=estimator,
        aggregation=aggregation,
        n_channels=n_channels,
        sfreq=sfreq,
        window_sec=window_sec,
        return_label_names=return_label_names,
    )


def fit_score_riemann(Ctr, ytr, Cev, yev, label_names=None, **kwargs):
    """Fit a Riemannian classifier and return eval metrics.

    Defaults to the previous tangent-space logistic probe, but accepts
    ``classifier="mdm"`` and metric options from ``fit_score_riemannian``.
    """
    return fit_score_riemannian(Ctr, ytr, Cev, yev, label_names=label_names, **kwargs)


def run_recording_riemann(
    data_root=None,
    label_scheme="tuab",
    class_names=None,
    train_split="train",
    eval_split="eval",
    n_windows=None,
    estimator="oas",
    n_channels=None,
    sfreq=None,
    window_sec=None,
    aggregation="riemann",
    classifier="tangent-logreg",
    mean_metric="riemann",
    distance_metric="riemann",
    tangent_metric="riemann",
    cov_regularization=1e-6,
    riemann_max_iter=50,
    riemann_tol=1e-7,
):
    """Read both splits, compute recording covariances, fit and score."""
    return run_recording_riemannian(
        data_root=data_root,
        label_scheme=label_scheme,
        class_names=class_names,
        train_split=train_split,
        eval_split=eval_split,
        n_windows=n_windows,
        estimator=estimator,
        aggregation=aggregation,
        classifier=classifier,
        mean_metric=mean_metric,
        distance_metric=distance_metric,
        tangent_metric=tangent_metric,
        n_channels=n_channels,
        sfreq=sfreq,
        window_sec=window_sec,
        cov_regularization=cov_regularization,
        riemann_max_iter=riemann_max_iter,
        riemann_tol=riemann_tol,
    )


def _parse_args(argv):
    return build_argparser().parse_args(argv)


def main():
    _riemannian_main(sys.argv[1:])


if __name__ == "__main__":
    main()
