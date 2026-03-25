#!/usr/bin/env python3
"""
AutoResearch Leaderboard — analyze experiment results.
Usage: uv run leaderboard.py [--format json|csv|md] [--top N]
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

RESULTS_TSV = "results.tsv"
RESULTS_DIR = "results"


def load_results(path=RESULTS_TSV):
    if not os.path.exists(path):
        print(f"No results file found at {path}")
        sys.exit(1)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                row["val_bpb"] = float(row["val_bpb"])
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def filter_kept(rows):
    return [r for r in rows if r.get("status") == "keep"]


def export_json(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(rows)} rows to {path}")


def export_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = rows[0].keys()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to {path}")


def print_markdown(rows, top_n=10):
    rows = sorted(rows, key=lambda r: r["val_bpb"])
    rows = rows[:top_n]
    print("| Rank | val_bpb | Status | Description | Commit |")
    print("|------|---------|--------|-------------|--------|")
    for i, r in enumerate(rows, 1):
        bpb = f"{r['val_bpb']:.6f}"
        status = r.get("status", "?")
        desc = r.get("description", "")[:50]
        commit = r.get("commit", "?")[:8]
        print(f"| {i} | {bpb} | {status} | {desc} | `{commit}` |")


def print_summary(rows):
    all_rows = rows
    kept = filter_kept(rows)
    crashes = [r for r in rows if r.get("status") == "crash"]
    discards = [r for r in rows if r.get("status") == "discard"]

    print("\n=== AutoResearch Leaderboard ===\n")
    print(f"Total experiments: {len(all_rows)}")
    print(f"  Kept (improved):   {len(kept)}")
    print(f"  Discarded:         {len(discards)}")
    print(f"  Crashes:           {len(crashes)}")

    if kept:
        best = min(kept, key=lambda r: r["val_bpb"])
        worst = max(kept, key=lambda r: r["val_bpb"])
        print(f"\nBest val_bpb:  {best['val_bpb']:.6f} ({best.get('description', '?')})")
        print(f"Worst kept:    {worst['val_bpb']:.6f}")
        print(f"Improvement:   {worst['val_bpb'] - best['val_bpb']:.6f} bpb")


def main():
    parser = argparse.ArgumentParser(description="AutoResearch Leaderboard")
    parser.add_argument("--format", choices=["json", "csv", "md"], default="md",
                        help="Output format (default: md)")
    parser.add_argument("--top", type=int, default=10, help="Top N results (default: 10)")
    parser.add_argument("--all", action="store_true", help="Include discarded/crashed runs")
    parser.add_argument("--export", action="store_true",
                        help=f"Export to {RESULTS_DIR}/ directory")
    args = parser.parse_args()

    rows = load_results()
    if not args.all:
        rows = filter_kept(rows)

    print_summary(load_results())

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.format == "json":
        top_rows = sorted(rows, key=lambda r: r["val_bpb"])[:args.top]
        if args.export:
            export_json(top_rows, os.path.join(RESULTS_DIR, f"leaderboard_{ts}.json"))
        else:
            print(json.dumps(top_rows, indent=2, ensure_ascii=False))
    elif args.format == "csv":
        top_rows = sorted(rows, key=lambda r: r["val_bpb"])[:args.top]
        if args.export:
            export_csv(top_rows, os.path.join(RESULTS_DIR, f"leaderboard_{ts}.csv"))
        else:
            if top_rows:
                keys = top_rows[0].keys()
                w = csv.DictWriter(sys.stdout, fieldnames=keys)
                w.writeheader()
                w.writerows(top_rows)
    else:
        print_markdown(rows, args.top)


if __name__ == "__main__":
    main()
