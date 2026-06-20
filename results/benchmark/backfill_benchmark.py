"""Backfill the JEPA arms in eeg_benchmark.{csv,json} from the verified per-seed eval
artifact (jepa_eval_seeds.json). Idempotent: only touches rows with status=='pending_training'.
Run from repo root:  python results/benchmark/backfill_benchmark.py
"""
import csv
import json
import os

HERE = os.path.dirname(__file__)
means = json.load(open(os.path.join(HERE, "jepa_eval_seeds.json"), encoding="utf-8"))["cells"]

# method-substring -> cell key in jepa_eval_seeds.json
MAP = {
    "VICReg ambient": "c0_vicreg_ambient",
    "SIGReg ambient": "c1_sigreg_ambient",
    "SIGReg tangent": "c2_sigreg_tangent",
    "PEIRA ambient": "c3_peira_ambient",
    "PEIRA tangent": "c4_peira_tangent",
}


def cell_for(method):
    for sub, key in MAP.items():
        if sub in (method or ""):
            return key
    return None


SEED = "1,1000,10000"
SRC = "local 3-seed mean (jepa_eval_seeds.json)"

# ---- CSV ----
csv_path = os.path.join(HERE, "eeg_benchmark.csv")
with open(csv_path, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    fields, rows = r.fieldnames, list(r)
n_csv = 0
for row in rows:
    if row.get("status") == "pending_training":
        c = cell_for(row.get("method"))
        if c:
            m = means[c]
            row["status"] = "measured_local"
            row["balanced_acc"] = f"{m['mean_balanced_acc']:.4f}"
            row["auroc"] = f"{m['mean_auroc']:.4f}"
            row["seed"] = SEED
            row["metric_source"] = SRC
            n_csv += 1
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)

# ---- JSON ----
json_path = os.path.join(HERE, "eeg_benchmark_rows.json")
data = json.load(open(json_path, encoding="utf-8"))
n_json = 0
for o in data:
    if o.get("status") == "pending_training":
        c = cell_for(o.get("method"))
        if c:
            m = means[c]
            o["status"] = "measured_local"
            o["balanced_acc"] = round(m["mean_balanced_acc"], 4)
            o["auroc"] = round(m["mean_auroc"], 4)
            o["seed"] = SEED
            o["metric_source"] = SRC
            n_json += 1
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")

print(f"backfilled CSV rows={n_csv}, JSON rows={n_json}")
