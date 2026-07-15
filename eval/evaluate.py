"""General evaluation entrypoint for segmentation and classification outputs.

This script evaluates saved predictions, regardless of which model produced
them. It intentionally keeps inference out of scope so CycleGAN, SSVS, ARCADE,
and future classifiers can share one metrics path after they write predictions.

Examples:
    python eval/evaluate.py segmentation --pred-dir pred --gt-dir masks
    python eval/evaluate.py segmentation --pred-dir pred --gt-coco datasets/arcade/val/annotations/val.json
    python eval/evaluate.py classification --pred-csv preds.csv --gt-csv labels.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def binary_confusion(pred, target):
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    return {
        "tp": int(np.count_nonzero(pred & target)),
        "tn": int(np.count_nonzero(~pred & ~target)),
        "fp": int(np.count_nonzero(pred & ~target)),
        "fn": int(np.count_nonzero(~pred & target)),
    }


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def segmentation_metrics(pred, target):
    counts = binary_confusion(pred, target)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    return {
        **counts,
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "iou": safe_divide(tp, tp + fp + fn),
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn),
    }


def summarize_metric_rows(rows, metric_names):
    summary = {"count": len(rows)}
    for name in metric_names:
        values = np.array([row[name] for row in rows], dtype=np.float64)
        summary[name] = {
            "mean": float(np.mean(values)) if len(values) else 0.0,
            "std": float(np.std(values)) if len(values) else 0.0,
        }
    return summary


def add_bootstrap_ci(summary, rows, metric_names, confidence_level=0.95, samples=2000, seed=13):
    if not rows:
        return summary

    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)
    row_count = len(rows)

    for name in metric_names:
        values = np.array([row[name] for row in rows], dtype=np.float64)
        bootstrap_means = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            sampled_indices = rng.integers(0, row_count, size=row_count)
            bootstrap_means[index] = float(np.mean(values[sampled_indices]))
        summary[name]["ci"] = {
            "level": confidence_level,
            "method": "case_bootstrap_mean",
            "samples": samples,
            "seed": seed,
            "lower": float(np.percentile(bootstrap_means, lower_percentile)),
            "upper": float(np.percentile(bootstrap_means, upper_percentile)),
        }
    return summary


def load_binary_image(path, threshold):
    image = Image.open(path).convert("L")
    array = np.array(image, dtype=np.float32)
    return array >= threshold


def image_files(path):
    return sorted(
        file_path
        for file_path in Path(path).iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def mask_from_polygon(segmentation, width, height):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentation:
        if not polygon:
            continue
        try:
            coords = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(coords, outline=1, fill=1)
        except (IndexError, TypeError, ValueError):
            continue
    return np.array(mask, dtype=bool)


def mask_from_rle(segmentation):
    try:
        from pycocotools import mask as coco_mask
    except ImportError as exc:
        raise ImportError(
            "COCO RLE masks require pycocotools. Install it or export masks as images."
        ) from exc
    return coco_mask.decode(segmentation).astype(bool)


def coco_masks_by_stem(coco_json, category_contains=None):
    with open(coco_json, "r", encoding="utf-8") as file:
        coco = json.load(file)

    categories = {category["id"]: category.get("name", "") for category in coco.get("categories", [])}
    selected_categories = set(categories)
    if category_contains:
        needles = [value.lower() for value in category_contains]
        selected_categories = {
            cat_id
            for cat_id, name in categories.items()
            if any(needle in str(name).lower() for needle in needles)
        }

    images = {image["id"]: image for image in coco.get("images", [])}
    masks = {}
    for image_id, image_info in images.items():
        stem = Path(image_info["file_name"]).stem
        masks[stem] = np.zeros((image_info["height"], image_info["width"]), dtype=bool)

    for annotation in coco.get("annotations", []):
        if annotation.get("category_id") not in selected_categories:
            continue
        image_info = images.get(annotation.get("image_id"))
        if not image_info:
            continue
        segmentation = annotation.get("segmentation")
        if not segmentation:
            continue
        if isinstance(segmentation, dict):
            mask = mask_from_rle(segmentation)
        else:
            mask = mask_from_polygon(segmentation, image_info["width"], image_info["height"])
        masks[Path(image_info["file_name"]).stem] |= mask

    return masks


def folder_masks_by_stem(gt_dir, threshold):
    return {
        file_path.stem: load_binary_image(file_path, threshold)
        for file_path in image_files(gt_dir)
    }


def write_rows_csv(path, rows):
    if not path or not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    if not path:
        return
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def prediction_id(pred_path, pred_suffix):
    stem = pred_path.stem
    return stem[: -len(pred_suffix)] if pred_suffix and stem.endswith(pred_suffix) else stem


def evaluate_segmentation(args):
    pred_paths = image_files(args.pred_dir)
    if args.pred_suffix:
        pred_paths = [path for path in pred_paths if path.stem.endswith(args.pred_suffix)]
    if args.gt_coco:
        gt_masks = coco_masks_by_stem(args.gt_coco, args.coco_category_contains)
    else:
        gt_masks = folder_masks_by_stem(args.gt_dir, args.gt_threshold)

    rows = []
    missing = []
    for pred_path in pred_paths:
        pred_mask = load_binary_image(pred_path, args.pred_threshold)
        sample_id = prediction_id(pred_path, args.pred_suffix)
        gt_mask = gt_masks.get(sample_id)
        if gt_mask is None:
            missing.append(pred_path.name)
            continue
        if pred_mask.shape != gt_mask.shape:
            pred_mask = np.array(
                Image.fromarray(pred_mask.astype(np.uint8)).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    Image.NEAREST,
                ),
                dtype=bool,
            )
        rows.append({"id": sample_id, **segmentation_metrics(pred_mask, gt_mask)})

    metric_names = ["precision", "recall", "specificity", "iou", "dice"]
    summary = summarize_metric_rows(rows, metric_names)
    if args.bootstrap_ci:
        summary = add_bootstrap_ci(
            summary,
            rows,
            metric_names,
            confidence_level=args.ci_level,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    summary["missing_ground_truth"] = missing
    payload = {"task": "segmentation", "summary": summary, "per_sample": rows}

    write_rows_csv(args.per_sample_csv, rows)
    write_json(args.output_json, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


def read_label_csv(path, id_column, label_column, score_column=None):
    rows = {}
    with open(path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            record = {"label": row[label_column]}
            if score_column:
                record["score"] = float(row[score_column])
            rows[row[id_column]] = record
    return rows


def classification_summary(rows):
    labels = sorted({row["gt"] for row in rows} | {row["pred"] for row in rows})
    per_class = {}
    correct = 0
    for label in labels:
        tp = sum(row["pred"] == label and row["gt"] == label for row in rows)
        fp = sum(row["pred"] == label and row["gt"] != label for row in rows)
        fn = sum(row["pred"] != label and row["gt"] == label for row in rows)
        support = sum(row["gt"] == label for row in rows)
        correct += tp
        precision_value = safe_divide(tp, tp + fp)
        recall_value = safe_divide(tp, tp + fn)
        per_class[label] = {
            "precision": precision_value,
            "recall": recall_value,
            "f1": safe_divide(2 * precision_value * recall_value, precision_value + recall_value),
            "support": support,
        }

    f1_values = [metrics["f1"] for metrics in per_class.values()]
    return {
        "count": len(rows),
        "accuracy": safe_divide(correct, len(rows)),
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def evaluate_classification(args):
    preds = read_label_csv(args.pred_csv, args.id_column, args.pred_column, args.score_column)
    labels = read_label_csv(args.gt_csv, args.id_column, args.gt_column)
    rows = []
    missing = []
    for sample_id, gt_record in labels.items():
        pred_record = preds.get(sample_id)
        if pred_record is None:
            missing.append(sample_id)
            continue
        row = {"id": sample_id, "gt": gt_record["label"], "pred": pred_record["label"]}
        if "score" in pred_record:
            row["score"] = pred_record["score"]
        rows.append(row)

    summary = classification_summary(rows)
    if args.bootstrap_ci and rows:
        summary = add_classification_bootstrap_ci(
            summary,
            rows,
            confidence_level=args.ci_level,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    summary["missing_predictions"] = missing
    payload = {"task": "classification", "summary": summary, "per_sample": rows}

    write_rows_csv(args.per_sample_csv, rows)
    write_json(args.output_json, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


def add_classification_bootstrap_ci(summary, rows, confidence_level=0.95, samples=2000, seed=13):
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence_level
    lower_percentile = 100.0 * alpha / 2.0
    upper_percentile = 100.0 * (1.0 - alpha / 2.0)
    row_count = len(rows)
    bootstrap_accuracy = np.empty(samples, dtype=np.float64)
    bootstrap_macro_f1 = np.empty(samples, dtype=np.float64)

    for index in range(samples):
        sampled_indices = rng.integers(0, row_count, size=row_count)
        sampled_rows = [rows[row_index] for row_index in sampled_indices]
        sampled_summary = classification_summary(sampled_rows)
        bootstrap_accuracy[index] = sampled_summary["accuracy"]
        bootstrap_macro_f1[index] = sampled_summary["macro_f1"]

    for name, values in {
        "accuracy": bootstrap_accuracy,
        "macro_f1": bootstrap_macro_f1,
    }.items():
        summary[name] = {
            "mean": float(summary[name]),
            "ci": {
                "level": confidence_level,
                "method": "case_bootstrap",
                "samples": samples,
                "seed": seed,
                "lower": float(np.percentile(values, lower_percentile)),
                "upper": float(np.percentile(values, upper_percentile)),
            },
        }
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate model predictions across tasks")
    subparsers = parser.add_subparsers(dest="task", required=True)

    segmentation = subparsers.add_parser("segmentation", help="Evaluate binary mask predictions")
    segmentation.add_argument("--pred-dir", required=True, help="Directory of predicted mask images")
    gt_group = segmentation.add_mutually_exclusive_group(required=True)
    gt_group.add_argument("--gt-dir", help="Directory of ground-truth mask images")
    gt_group.add_argument("--gt-coco", help="COCO annotation JSON for ground-truth masks")
    segmentation.add_argument("--pred-threshold", type=float, default=128.0)
    segmentation.add_argument("--gt-threshold", type=float, default=1.0)
    segmentation.add_argument(
        "--pred-suffix",
        default="",
        help="Suffix to strip from prediction filenames before matching labels, e.g. _vessel_pred",
    )
    segmentation.add_argument(
        "--coco-category-contains",
        action="append",
        help="Only use COCO categories whose names contain this text; repeatable",
    )
    segmentation.add_argument("--output-json", default="")
    segmentation.add_argument("--per-sample-csv", default="")
    segmentation.add_argument("--bootstrap-ci", action="store_true", help="Add case-level bootstrap confidence intervals")
    segmentation.add_argument("--ci-level", type=float, default=0.95, help="Confidence level for bootstrap intervals")
    segmentation.add_argument("--bootstrap-samples", type=int, default=2000, help="Number of bootstrap resamples")
    segmentation.add_argument("--bootstrap-seed", type=int, default=13, help="Random seed for bootstrap resampling")
    segmentation.set_defaults(func=evaluate_segmentation)

    classification = subparsers.add_parser("classification", help="Evaluate classification CSVs")
    classification.add_argument("--pred-csv", required=True)
    classification.add_argument("--gt-csv", required=True)
    classification.add_argument("--id-column", default="id")
    classification.add_argument("--pred-column", default="pred")
    classification.add_argument("--gt-column", default="label")
    classification.add_argument("--score-column", default="")
    classification.add_argument("--output-json", default="")
    classification.add_argument("--per-sample-csv", default="")
    classification.add_argument("--bootstrap-ci", action="store_true", help="Add case-level bootstrap confidence intervals")
    classification.add_argument("--ci-level", type=float, default=0.95, help="Confidence level for bootstrap intervals")
    classification.add_argument("--bootstrap-samples", type=int, default=2000, help="Number of bootstrap resamples")
    classification.add_argument("--bootstrap-seed", type=int, default=13, help="Random seed for bootstrap resampling")
    classification.set_defaults(func=evaluate_classification)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "score_column") and not args.score_column:
        args.score_column = None
    args.func(args)


if __name__ == "__main__":
    main()
