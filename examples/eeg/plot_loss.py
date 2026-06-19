"""Compare training loss curves between TUAB and TUEV runs.

Usage (local, after fetching logs from Dalia):
  python -m examples.eeg.plot_loss \
    --tuab  checkpoints/eeg_ambient_sigreg/train_log.json \
    --tuev  checkpoints/eeg_tuev_sigreg/train_log.json

Also parses old-style stdout logs (train_*.out) when --tuab-log is given:
  python -m examples.eeg.plot_loss --tuab-log /path/to/train_74706.out --tuev ...
"""
import argparse
import json
import re
import sys


def load_json_log(path):
    with open(path) as f:
        rows = json.load(f)
    return [r["epoch"] for r in rows], [r["loss"] for r in rows]


def parse_stdout_log(path):
    epochs, losses = [], []
    pat = re.compile(r"\[eeg\] epoch\s+(\d+)\s+loss=([\d.]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return epochs, losses


def ascii_chart(series: dict, width=60, height=20):
    """Render multiple (epochs, losses) series as an ASCII chart."""
    all_losses = [v for _, ls in series.values() for v in ls]
    if not all_losses:
        return "(no data)"
    lo, hi = min(all_losses), max(all_losses)
    if hi == lo:
        hi = lo + 1
    max_epoch = max(max(ep) for ep, _ in series.values())

    symbols = {"TUAB": "█", "TUEV": "▲"}
    canvas = [[" "] * width for _ in range(height)]

    for label, (epochs, losses) in series.items():
        sym = symbols.get(label, "●")
        for ep, lo_ in zip(epochs, losses):
            x = int(ep / max_epoch * (width - 1))
            y = int((hi - lo_) / (hi - lo) * (height - 1))
            y = max(0, min(height - 1, y))
            canvas[y][x] = sym

    lines = []
    for i, row in enumerate(canvas):
        yval = hi - (i / (height - 1)) * (hi - lo)
        lines.append(f"{yval:7.4f} |{''.join(row)}")
    lines.append("        +" + "-" * width)
    lines.append(f"         0{' ' * (width // 2 - 5)}epoch{' ' * (width // 2 - 5)}{max_epoch}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tuab", help="TUAB train_log.json")
    p.add_argument("--tuab-log", help="TUAB stdout log (train_*.out)")
    p.add_argument("--tuev", help="TUEV train_log.json")
    p.add_argument("--tuev-log", help="TUEV stdout log")
    args = p.parse_args()

    series = {}

    if args.tuab:
        series["TUAB"] = load_json_log(args.tuab)
    elif args.tuab_log:
        series["TUAB"] = parse_stdout_log(args.tuab_log)

    if args.tuev:
        series["TUEV"] = load_json_log(args.tuev)
    elif args.tuev_log:
        series["TUEV"] = parse_stdout_log(args.tuev_log)

    if not series:
        print("No logs provided. Use --tuab / --tuev with a .json or --tuab-log / --tuev-log with a .out file.")
        sys.exit(1)

    for label, (epochs, losses) in series.items():
        print(f"\n{label}: {len(losses)} epochs  "
              f"loss_start={losses[0]:.4f}  loss_end={losses[-1]:.4f}  "
              f"drop={losses[0]-losses[-1]:.4f} ({(1-losses[-1]/losses[0])*100:.1f}%)")

    print("\n=== Loss curve (█=TUAB  ▲=TUEV) ===")
    print(ascii_chart(series))

    if "TUAB" in series and "TUEV" in series:
        tuab_drop = series["TUAB"][1][0] - series["TUAB"][1][-1]
        tuev_drop = series["TUEV"][1][0] - series["TUEV"][1][-1]
        print(f"\nTUAB loss drop : {tuab_drop:.4f}")
        print(f"TUEV loss drop : {tuev_drop:.4f}")
        print("\n(Plus le drop est grand → le SSL a appris quelque chose)")


if __name__ == "__main__":
    main()
