#!/usr/bin/env python3
"""
Compute accuracy + calibration metrics (ECE, Brier, SPR) from chemistry MCQ inference output.

Usage:
  python eval/run_eval_chemistry.py --input outputs/chemistry_inference.json --output outputs/chemistry_metrics.json
"""
import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np

from calibration_metrics import get_brier, get_ece, get_spr, get_overconfidence_gap, bin_stats
from parse_confidence import confidence_extractor
from chemistry_correctness import check_correctness_one


def parse_args():
    p = argparse.ArgumentParser(description="Compute calibration metrics from chemistry inference JSON")
    p.add_argument("--input", type=str, required=True, help="JSON from inference")
    p.add_argument("--output", type=str, default=None, help="Write metrics JSON here")
    p.add_argument("--n_bins", type=int, default=10, help="ECE bins")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.input) as f:
        data = json.load(f)

    correctness_list = []
    confidence_list = []
    format_ok_list = []
    parsed_answers = []

    for item in data:
        response = item.get("response", "")
        golden = item.get("golden_answer", [])
        correct, parsed = check_correctness_one(response, golden)
        correctness_list.append(1.0 if correct else 0.0)
        parsed_answers.append(parsed)
        fmt_ok, conf = confidence_extractor(response)
        format_ok_list.append(fmt_ok)
        confidence_list.append(conf)

    correctness = np.array(correctness_list, dtype=np.float64)
    confidence = np.array(confidence_list, dtype=np.float64)
    format_ok = np.array(format_ok_list, dtype=np.int32) == 1

    accuracy = float(np.mean(correctness))
    brier = get_brier(correctness, confidence)
    ece = get_ece(correctness, confidence, n_bins=args.n_bins)
    spr = get_spr(correctness, confidence)
    ocg = get_overconfidence_gap(correctness, confidence)
    format_adherence = float(np.mean(format_ok))
    mean_conf = float(np.mean(confidence))
    n_parsed = sum(1 for a in parsed_answers if a is not None)
    answer_parse_rate = n_parsed / len(data) if data else 0.0

    metrics = {
        "accuracy": accuracy,
        "answer_parse_rate": answer_parse_rate,
        "brier_score": brier,
        "ece": ece,
        "spr": spr,
        "overconfidence_gap": ocg,
        "mean_confidence": mean_conf,
        "confidence_format_adherence": format_adherence,
        "n_samples": len(correctness),
        "n_bins": args.n_bins,
    }
    metrics["bin_stats"] = bin_stats(correctness, confidence, n_bins=args.n_bins)

    n_valid = int(np.sum(format_ok))
    if n_valid >= 2:
        c_valid = correctness[format_ok]
        conf_valid = confidence[format_ok]
        metrics["valid_confidence_only"] = {
            "n_valid": n_valid,
            "accuracy": float(np.mean(c_valid)),
            "brier_score": get_brier(c_valid, conf_valid),
            "ece": get_ece(c_valid, conf_valid, n_bins=args.n_bins),
            "spr": get_spr(c_valid, conf_valid),
            "mean_confidence": float(np.mean(conf_valid)),
        }

    print("Chemistry MCQ calibration metrics:")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Answer parse rate: {answer_parse_rate:.2%}")
    print(f"  Brier Score:       {brier:.4f}")
    print(f"  ECE:               {ece:.4f}")
    print(f"  SPR:               {spr:.4f}")
    print(f"  Overconf. Gap:     {ocg:+.4f}")
    print(f"  Mean confidence:   {mean_conf:.4f}")
    print(f"  Format adherence:  {format_adherence:.2%}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_metrics = {k: v for k, v in metrics.items() if k != "bin_stats"}
        out_metrics["bin_stats"] = [
            {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in b.items()}
            for b in metrics["bin_stats"]
        ]
        if "valid_confidence_only" in metrics:
            out_metrics["valid_confidence_only"] = metrics["valid_confidence_only"]
        with open(out_path, "w") as f:
            json.dump(out_metrics, f, indent=2)
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
