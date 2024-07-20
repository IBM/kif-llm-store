# Copyright (C) 2024 IBM Corp.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import pathlib
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd


def read_jsonl_file(file_path: Union[str, Path]) -> List[Dict]:
    with open(file_path, "r") as f:
        rows = [json.loads(line) for line in f]
    return rows


def true_positives(preds: List, gts: List) -> int:
    tp = 0
    for pred in preds:
        if pred in gts:
            tp += 1

    return tp


def precision(preds: List[str], gts: List[str]) -> float:
    try:
        # When nothing is predicted, precision = 1
        # irrespective of the ground truth value
        if len(preds) == 0:
            return 1
        # When the predictions are not empty
        return min(true_positives(preds, gts) / len(preds), 1.0)
    except TypeError:
        return 0.0


def recall(preds: List[str], gts: List[str]) -> float:
    try:
        # When ground truth is empty return 1
        # even if there are predictions (edge case)
        if len(gts) == 0:
            return 1.0
        # When the ground truth is not empty
        return min(true_positives(preds, gts) / len(gts), 1.0)
    except TypeError:
        return 0.0


def f1_score(p: float, r: float) -> float:
    try:
        return (2 * p * r) / (p + r)
    except ZeroDivisionError:
        return 0.0


def rows_to_dict(rows: List[Dict]) -> Dict:
    """Index the ground truth/prediction rows by subject entity and
    relation."""
    return {
        (r["SubjectEntity"], r["Relation"]): list(set(r["ObjectEntitiesID"]))
        for r in rows
    }


def evaluate_per_sr_pair(pred_rows, gt_rows) -> List[Dict[str, float]]:
    """Evaluate the predictions per Subject-Relation pair"""
    pred_dict = rows_to_dict(pred_rows)
    gt_dict = rows_to_dict(gt_rows)

    results = []

    for subj, rel in gt_dict:
        # get the ground truth objects
        gts = gt_dict[(subj, rel)]

        # get the predictions
        preds = pred_dict[(subj, rel)]

        # calculate the scores
        p = precision(preds, gts)
        r = recall(preds, gts)
        f1 = f1_score(p, r)

        results.append(
            {
                "SubjectEntity": subj,
                "Relation": rel,
                "p": p,
                "r": r,
                "f1": f1,
                "tp": true_positives(preds, gts),
                "total_pred": len(preds),
                "total_gt": len(gts),
            }
        )

    return sorted(results, key=lambda x: (x["Relation"], x["SubjectEntity"]))


def macro_average_per_relation(scores_per_sr: List[Dict[str, float]]) -> dict:
    """Compute the macro average scores per relation"""
    scores = {}
    for r in scores_per_sr:
        if r["Relation"] not in scores:
            scores[r["Relation"]] = []
        scores[r["Relation"]].append(
            {
                "p": r["p"],
                "r": r["r"],
                "f1": r["f1"],
            }
        )

    macro_averages = {}
    for rel in scores:
        macro_averages[rel] = {
            "macro-p": sum([x["p"] for x in scores[rel]]) / len(scores[rel]),
            "macro-r": sum([x["r"] for x in scores[rel]]) / len(scores[rel]),
            "macro-f1": sum([x["f1"] for x in scores[rel]]) / len(scores[rel]),
        }

    # Macro average for all relations
    all_rel_macro_p = sum([x["p"] for x in scores_per_sr]) / len(scores_per_sr)
    all_rel_macro_r = sum([x["r"] for x in scores_per_sr]) / len(scores_per_sr)
    all_rel_macro_f1 = sum([x["f1"] for x in scores_per_sr]) / len(
        scores_per_sr
    )

    macro_averages["*** All Relations ***"] = {
        "macro-p": all_rel_macro_p,
        "macro-r": all_rel_macro_r,
        "macro-f1": all_rel_macro_f1,
    }

    return macro_averages


def micro_average_per_relation(scores_per_sr: List[Dict[str, float]]) -> dict:
    """Compute the micro average scores per relation"""
    scores = {}
    for r in scores_per_sr:
        if r["Relation"] not in scores:
            scores[r["Relation"]] = {
                "tp": 0,
                "total_pred": 0,
                "total_gt": 0,
            }
        scores[r["Relation"]]["tp"] += r["tp"]
        scores[r["Relation"]]["total_pred"] += r["total_pred"]
        scores[r["Relation"]]["total_gt"] += r["total_gt"]

    micro_averages = {}
    for rel in scores:
        micro_p = (
            scores[rel]["tp"] / scores[rel]["total_pred"]
            if scores[rel]["total_pred"] > 0
            else 1.0
        )
        micro_r = (
            scores[rel]["tp"] / scores[rel]["total_gt"]
            if scores[rel]["total_gt"] > 0
            else 1.0
        )

        micro_averages[rel] = {
            "micro-p": micro_p,
            "micro-r": micro_r,
            "micro-f1": f1_score(micro_p, micro_r),
        }

    # Micro average for all relations
    total_tp = sum([x["tp"] for x in scores.values()])
    total_pred = sum([x["total_pred"] for x in scores.values()])
    total_gt = sum([x["total_gt"] for x in scores.values()])

    all_rel_micro_p = total_tp / total_pred if total_pred > 0 else 1.0
    all_rel_micro_r = total_tp / total_gt if total_gt > 0 else 1.0

    micro_averages["*** All Relations ***"] = {
        "micro-p": all_rel_micro_p,
        "micro-r": all_rel_micro_r,
        "micro-f1": f1_score(all_rel_micro_p, all_rel_micro_r),
    }

    return micro_averages


def prediction_statistics(scores_per_sr: List[Dict[str, float]]) -> dict:
    """Get the average numbers of predictions and the numbers of
    empty predictions per relation."""
    stats = {}
    for r in scores_per_sr:
        if r["Relation"] not in stats:
            stats[r["Relation"]] = {
                "num_sr_pairs": 0,
                "total_pred": 0,
                "empty_pred": 0,
            }
        stats[r["Relation"]]["num_sr_pairs"] += 1
        stats[r["Relation"]]["total_pred"] += r["total_pred"]
        if r["total_pred"] == 0:
            stats[r["Relation"]]["empty_pred"] += 1

    final_stats = {}
    for rel in stats:
        final_stats[rel] = {
            "avg. #preds": stats[rel]["total_pred"]
            / stats[rel]["num_sr_pairs"],
            "#empty preds": stats[rel]["empty_pred"],
        }

    # Average numbers of predictions and the numbers of empty
    # predictions for all relations
    total_sr_pairs = len(scores_per_sr)
    total_preds = sum([x["total_pred"] for x in stats.values()])
    total_empty_preds = sum([x["empty_pred"] for x in stats.values()])

    final_stats["*** All Relations ***"] = {
        "avg. #preds": total_preds / total_sr_pairs,
        "#empty preds": total_empty_preds,
    }

    return final_stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Precision, Recall and F1-score of predictions"
    )

    parser.add_argument(
        "-p",
        "--predictions",
        type=str,
        required=True,
        help="Path to the predictions file (required)",
    )
    parser.add_argument(
        "-g",
        "--ground_truth",
        type=str,
        required=True,
        help="Path to the ground truth file (required)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file",
    )

    args = parser.parse_args()

    # Read the predictions and ground truth
    pred_rows = read_jsonl_file(args.predictions)
    gt_rows = read_jsonl_file(args.ground_truth)

    output_file_name = os.path.splitext(os.path.basename(args.predictions))[0]
    output_file = (
        args.output
        if args.output
        else (
            f'{pathlib.Path(__file__).parent.resolve()}/lm-kbc-analysis/output/'
            f'evaluation_results/{output_file_name}.csv'
        )
    )

    # Evaluate the predictions
    scores_per_sr_pair = evaluate_per_sr_pair(pred_rows, gt_rows)

    # Macro average
    macro_per_relation = macro_average_per_relation(scores_per_sr_pair)
    macro_df = pd.DataFrame(macro_per_relation).transpose().round(3)

    # Micro average
    micro_per_relation = micro_average_per_relation(scores_per_sr_pair)
    micro_df = pd.DataFrame(micro_per_relation).transpose().round(3)

    # Statistics
    stats = prediction_statistics(scores_per_sr_pair)
    stats_df = pd.DataFrame(stats).transpose().round(3)
    stats_df["#empty preds"] = stats_df["#empty preds"].astype(int)

    # Combine the results
    results = pd.concat([macro_df, micro_df, stats_df], axis=1)
    print(results)

    results.to_csv(output_file, index=True)


if __name__ == "__main__":
    main()
