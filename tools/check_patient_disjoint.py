#!/usr/bin/env python3
"""Assert TUAB-style train/eval folders are patient-disjoint (committed leakage gate).

Every probe number (`eval.py`, `label_efficiency.py`, `baseline_riemann.py`) assumes
the ``train/`` and ``eval/`` recordings come from disjoint patients. TUH/TUAB is
disjoint by construction, but nothing in the pipeline checks it — a re-export that
mixed patients would silently inflate every reported number. This is that check,
committed so the "2076 vs 253 patients, overlap 0" claim is reproducible from the
repo rather than only from a one-off interactive run on the cluster.

Patient ID = the first ``_``-delimited token of the EDF basename (TUH convention
``aaaaaaaa_s001_t000.edf`` -> patient ``aaaaaaaa``).

Usage:
  python -m tools.check_patient_disjoint --data-root <TUAB_PREPROCESSED>
  python -m tools.check_patient_disjoint --data-root <ROOT> --splits train eval

Exit code 0 if all splits are pairwise patient-disjoint, 1 otherwise.
"""
import argparse
import glob
import os
import sys


def patient_id(edf_path):
    return os.path.basename(edf_path).split("_")[0]


def split_patients(root, split):
    files = glob.glob(os.path.join(root, split, "**", "*.edf"), recursive=True)
    return {patient_id(f) for f in files}, len(files)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "eval"])
    a = ap.parse_args(argv)

    pats = {}
    for s in a.splits:
        p, n = split_patients(a.data_root, s)
        if n == 0:
            sys.exit(f"[disjoint] no .edf under {os.path.join(a.data_root, s)}")
        pats[s] = p
        print(f"[disjoint] {s}: {n} recordings, {len(p)} patients", flush=True)

    bad = False
    splits = list(pats)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            inter = pats[splits[i]] & pats[splits[j]]
            if inter:
                bad = True
                print(f"[disjoint] OVERLAP {splits[i]} & {splits[j]}: "
                      f"{len(inter)} shared patients e.g. {sorted(inter)[:5]}")
            else:
                print(f"[disjoint] OK {splits[i]} & {splits[j]}: overlap 0")

    if bad:
        sys.exit("[disjoint] FAIL: splits are NOT patient-disjoint — probe numbers leak")
    print("[disjoint] PASS: all splits patient-disjoint")


if __name__ == "__main__":
    main()
