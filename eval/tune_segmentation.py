"""Tune segmentation checkpoint and threshold on a validation set.

This script ranks existing prediction folders, one per checkpoint/epoch, across
one or more thresholds. It is intended for validation-only tuning before a
locked, one-time clinical test-set evaluation.

Example:
    python eval/tune_segmentation.py \
      --results-root results/arcade_unet_ngf32_true_combo_lr1e4 \
      --gt-coco datasets/arcade/val/annotations/val.json \
      --epochs 10 15 20 25 30 \
      --thresholds 144 152 160 168 176 \
      --pred-suffix _vessel_pred \
      --min-specificity 0.99 \
      --min-recall 0.72 \
      --bootstrap-ci \
      --output-json results/arcade_unet_ngf32_true_combo_lr1e4/tuning_report.json
"""

import argparse
import csv
import json
from pathlib import Path

from evaluate import (
    add_bootstrap_ci,
    coco_masks_by_stem,
    folder_masks_by_stem,
    image_files,
    load_binary_image,
    prediction_id,
    segmentation_metrics,
    summarize_metric_rows,
)


METRIC_NAMES = ["precision", "recall", "specificity", "iou", "dice"]


def parse_values(values, value_type):
    parsed = []
    for value in values:
        for part in str(value).split(","):
            if part.strip():
                parsed.append(value_type(part))
    return parsed


def evaluate_prediction_dir(pred_dir, gt_masks, pred_suffix, pred_threshold):
    pred_paths = image_files(pred_dir)
    if pred_suffix:
        pred_paths = [path for path in pred_paths if path.stem.endswith(pred_suffix)]

    rows = []
    missing = []
    for pred_path in pred_paths:
        sample_id = prediction_id(pred_path, pred_suffix)
        gt_mask = gt_masks.get(sample_id)
        if gt_mask is None:
            missing.append(pred_path.name)
            continue
        pred_mask = load_binary_image(pred_path, pred_threshold)
        if pred_mask.shape != gt_mask.shape:
            from PIL import Image
            import numpy as np

            pred_mask = np.array(
                Image.fromarray(pred_mask.astype("uint8")).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    Image.NEAREST,
                ),
                dtype=bool,
            )
        rows.append({"id": sample_id, **segmentation_metrics(pred_mask, gt_mask)})
    return rows, missing


def objective_value(summary, objective):
    if objective == "f1":
        return summary["dice"]["mean"]
    return summary[objective]["mean"]


def passes_constraints(summary, args):
    constraints = {
        "precision": args.min_precision,
        "recall": args.min_recall,
        "specificity": args.min_specificity,
        "iou": args.min_iou,
        "dice": args.min_dice,
    }
    for metric, minimum in constraints.items():
        if minimum is not None and summary[metric]["mean"] < minimum:
            return False
    return True


def candidate_row(candidate):
    summary = candidate["summary"]
    metric_values = {
        name: summary[name]["mean"] if name in summary else ""
        for name in METRIC_NAMES
    }
    return {
        "rank": candidate["rank"],
        "eligible": candidate["eligible"],
        "epoch": candidate["epoch"],
        "threshold": candidate["threshold"],
        "objective": candidate["objective"],
        **metric_values,
        "count": summary["count"],
        "missing_prediction_dir": candidate.get("missing_prediction_dir", ""),
    }


def write_csv(path, candidates):
    if not path:
        return
    rows = [candidate_row(candidate) for candidate in candidates]
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser():
    parser = argparse.ArgumentParser(description="Tune segmentation epoch and threshold on validation predictions")
    parser.add_argument("--results-root", required=True, help="Run results directory containing val_<epoch>/images folders")
    parser.add_argument("--phase-prefix", default="val", help="Prediction folder prefix, e.g. val for val_20")
    parser.add_argument("--epochs", nargs="+", required=True, help="Epochs to evaluate; accepts spaces or comma-separated values")
    parser.add_argument("--thresholds", nargs="+", required=True, help="Thresholds to evaluate; accepts spaces or comma-separated values")
    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument("--gt-dir", help="Directory of ground-truth masks")
    gt_group.add_argument("--gt-coco", help="COCO annotation JSON for ground-truth masks")
    parser.add_argument("--gt-threshold", type=float, default=1.0)
    parser.add_argument("--pred-suffix", default="_vessel_pred")
    parser.add_argument("--objective", choices=["dice", "iou", "precision", "recall", "specificity", "f1"], default="dice")
    parser.add_argument("--min-precision", type=float)
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--min-specificity", type=float)
    parser.add_argument("--min-iou", type=float)
    parser.add_argument("--min-dice", type=float)
    parser.add_argument("--bootstrap-ci", action="store_true", help="Add case-level bootstrap CIs to the selected candidate")
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    return parser


def main():
    args = build_parser().parse_args()
    epochs = parse_values(args.epochs, int)
    thresholds = parse_values(args.thresholds, float)
    results_root = Path(args.results_root)

    if args.gt_coco:
        gt_masks = coco_masks_by_stem(args.gt_coco)
        gt_source = args.gt_coco
    else:
        gt_masks = folder_masks_by_stem(args.gt_dir, args.gt_threshold)
        gt_source = args.gt_dir

    candidates = []
    for epoch in epochs:
        pred_dir = results_root / f"{args.phase_prefix}_{epoch}" / "images"
        if not pred_dir.is_dir():
            candidates.append({
                "epoch": epoch,
                "threshold": None,
                "eligible": False,
                "objective": 0.0,
                "summary": {"count": 0},
                "missing_prediction_dir": str(pred_dir),
            })
            continue

        for threshold in thresholds:
            rows, missing = evaluate_prediction_dir(pred_dir, gt_masks, args.pred_suffix, threshold)
            summary = summarize_metric_rows(rows, METRIC_NAMES)
            summary["missing_ground_truth"] = missing
            eligible = passes_constraints(summary, args)
            candidates.append({
                "epoch": epoch,
                "threshold": threshold,
                "eligible": eligible,
                "objective": objective_value(summary, args.objective) if rows else 0.0,
                "summary": summary,
            })

    ranked = sorted(
        candidates,
        key=lambda item: (item["eligible"], item["objective"]),
        reverse=True,
    )
    for index, candidate in enumerate(ranked, start=1):
        candidate["rank"] = index

    selected = ranked[0] if ranked else None
    if selected and args.bootstrap_ci and selected.get("summary", {}).get("count", 0):
        pred_dir = results_root / f"{args.phase_prefix}_{selected['epoch']}" / "images"
        rows, _ = evaluate_prediction_dir(pred_dir, gt_masks, args.pred_suffix, selected["threshold"])
        selected["summary"] = add_bootstrap_ci(
            selected["summary"],
            rows,
            METRIC_NAMES,
            confidence_level=args.ci_level,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )

    report = {
        "tuning_set": {
            "results_root": str(results_root),
            "ground_truth": gt_source,
            "phase_prefix": args.phase_prefix,
            "epochs": epochs,
            "thresholds": thresholds,
            "pred_suffix": args.pred_suffix,
        },
        "selection_rule": {
            "objective": args.objective,
            "constraints": {
                "min_precision": args.min_precision,
                "min_recall": args.min_recall,
                "min_specificity": args.min_specificity,
                "min_iou": args.min_iou,
                "min_dice": args.min_dice,
            },
        },
        "selected": selected,
        "top_candidates": ranked[: args.top_k],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")

    write_csv(args.output_csv, ranked)
    print(json.dumps(report["selected"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
