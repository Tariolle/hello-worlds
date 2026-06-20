#!/usr/bin/env python3
"""Build a patient-disjoint TUSZ SSL file_list for general-pretrain -> frozen TUAB.

The "apples-to-apples" jury claim (pretrain on general TUH = TUSZ, freeze, probe
TUAB) is only fair if NO TUSZ pretraining recording belongs to a patient in the
TUAB EVAL split — otherwise the 0.814 number is a patient leak. `cluster/
tusz_general.sbatch` reads `cfg.data.file_list` to restrict SSL to a path list, but
nothing in the repo built that list or proved the exclusion. This script does both:
it writes the TUSZ EDF paths whose patient ID is NOT in the TUAB-eval patient set,
and HARD-ASSERTS the result is disjoint from TUAB-eval (and prints the dropped
count for the job log). Point `cfg.data.file_list` at the output.

Patient ID = the first ``_``-delimited token of the EDF basename (TUH convention).

Usage:
  python -m tools.build_tusz_filelist \
      --tusz-root <TUSZ_PREPROCESSED> \
      --tuab-root <TUAB_PREPROCESSED> \
      --out tusz_ex_tuabeval.filelist
"""
import argparse
import glob
import os
import sys


def patient_id(p):
    return os.path.basename(p).split("_")[0]


def edfs(root, *parts):
    return glob.glob(os.path.join(root, *parts, "**", "*.edf"), recursive=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tusz-root", required=True, help="TUSZ corpus root (pretrain source)")
    ap.add_argument("--tuab-root", required=True, help="TUAB root (to read the eval split)")
    ap.add_argument("--tuab-eval-split", default="eval")
    ap.add_argument("--out", required=True, help="output newline-delimited file_list path")
    a = ap.parse_args(argv)

    tuab_eval = edfs(a.tuab_root, a.tuab_eval_split)
    if not tuab_eval:
        sys.exit(f"[tusz-filelist] no .edf under {os.path.join(a.tuab_root, a.tuab_eval_split)}")
    excl = {patient_id(f) for f in tuab_eval}
    print(f"[tusz-filelist] TUAB-{a.tuab_eval_split}: {len(tuab_eval)} recordings, "
          f"{len(excl)} patients to exclude", flush=True)

    tusz = sorted(edfs(a.tusz_root))
    if not tusz:
        sys.exit(f"[tusz-filelist] no .edf under {a.tusz_root}")
    kept = [f for f in tusz if patient_id(f) not in excl]
    print(f"[tusz-filelist] TUSZ: {len(tusz)} recordings -> kept {len(kept)}, "
          f"dropped {len(tusz) - len(kept)} (belonging to TUAB-eval patients)", flush=True)

    # Hard guarantee: the kept set shares no patient with TUAB-eval.
    overlap = {patient_id(f) for f in kept} & excl
    assert not overlap, (
        f"FAIL: {len(overlap)} TUAB-eval patients leaked into the list: {sorted(overlap)[:5]}")
    if not kept:
        sys.exit("[tusz-filelist] every TUSZ recording was excluded — check the roots/IDs")

    with open(a.out, "w") as fh:
        fh.write("\n".join(kept) + "\n")
    print(f"[tusz-filelist] wrote {len(kept)} paths to {a.out} "
          f"(patient-disjoint from TUAB-{a.tuab_eval_split})", flush=True)


if __name__ == "__main__":
    main()
