"""Aggregate only small E1 JSON summaries; never load datasets or models."""
import argparse
import csv
import json
from pathlib import Path
import statistics

METRICS = {"WMAPE_pct": "WMAPE_pct", "RMSE": "RMSE", "R2": "R2", "RelCon": "RelCon",
           "SUEGap": "SUEGap", "TSTT_Error_pct": "TSTT_Error_pct", "runtime_ms_per_graph": "Runtime_ms_graph"}
NETWORKS = ("siouxfalls", "ema", "anaheim")
METHODS = ("persistence", "od_only", "old_state", "hybrid")


def summarize(root, allow_partial=False):
    root = Path(root)
    rows, groups, missing = [], {}, []
    for network in NETWORKS:
        fingerprint, batch_size, device, protocol = None, None, None, None
        for method in METHODS:
            records = []
            seeds = [None] if method == "persistence" else [42, 43, 44, 45, 46]
            for seed in seeds:
                directory = root / network / method
                if seed is not None:
                    directory /= f"seed_{seed}"
                path = directory / "summary.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                with path.open(encoding="utf-8") as handle:
                    record = json.load(handle)
                if record["network"] != network or record["information_mode"] != method or record["seed"] != seed:
                    raise ValueError(f"Summary identity mismatch: {path}")
                if record.get("smoke") or record.get("formal_test_metric_passes") != 1:
                    raise ValueError(f"Not an official once-only test result: {path}")
                identity = (record["dataset_fingerprint"], record["runtime_batch_size"], record["device"], record["protocol_fingerprint"])
                if fingerprint is not None and identity != (fingerprint, batch_size, device, protocol):
                    raise ValueError(f"Different data/batch/device within {network}; cannot aggregate.")
                fingerprint, batch_size, device, protocol = identity
                records.append(record)
            if not records:
                continue
            row = {"Network": network, "Method": method, "N_runs": len(records)}
            for metric, label in METRICS.items():
                values = [record[metric] for record in records]
                row[label + "_mean"] = statistics.mean(values)
                row[label + "_std"] = statistics.stdev(values) if len(values) > 1 else None
            rows.append(row)
            groups[f"{network}/{method}"] = {"seeds": [r["seed"] for r in records],
                                            "mutation_type_breakdown": {r["seed"] if r["seed"] is not None else "persistence": r["mutation_type_breakdown"] for r in records}}
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} of 48 results. Use --allow-partial only for a clearly incomplete summary.")
    root.mkdir(parents=True, exist_ok=True)
    columns = ["Network", "Method", "N_runs"] + [label + suffix for label in METRICS.values() for suffix in ("_mean", "_std")]
    with (root / "e1_information_set_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "e1_information_set_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"complete": not missing, "missing_runs": missing, "rows": rows, "runs": groups,
                   "std_definition": "sample standard deviation (ddof=1); null for a single run"}, handle, indent=2, allow_nan=False)
    for row in rows:
        print(f"{row['Network']:12} {row['Method']:12} n={row['N_runs']} WMAPE={row['WMAPE_pct_mean']:.3f}% RMSE={row['RMSE_mean']:.3f}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/e1"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summarize(args.root, args.allow_partial)
