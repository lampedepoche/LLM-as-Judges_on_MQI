"""
gwet.py

Compute Gwet's AC1/AC2 (via irrCAC) for:
  - Human panel: all human annotators
  - Model substitution panels: (n-1) humans + model (all combinations), averaged

This is intended as a fast, descriptive companion to Krippendorff α results.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import itertools

from irrCAC.raw import CAC

# Import utility functions
from qualitative_analysis.metrics.utils import ensure_numeric_columns


def compute_gwet_panel_difference(
    detailed_results_df: pd.DataFrame,
    annotation_columns: List[str],
    model_column: str = "ModelPrediction",
    # "identity" -> AC1 ; "linear"/"quadratic"/custom matrix -> AC2
    ac_weights: str | np.ndarray = "identity",
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute Gwet AC (AC1/AC2) human-panel vs model-substitution-panel differences, per run,
    then aggregate across runs within each (prompt_name, iteration) scenario.

    Returns per-run point estimates only:
      - ac_human_groups
      - ac_model_groups (mean over substitution panels)
      - difference = ac_model_groups - ac_human_groups
    """
    # Ensure numeric columns (keeps NaN for missing ratings)
    columns_to_check = annotation_columns + [model_column]
    detailed_results_df = ensure_numeric_columns(detailed_results_df, columns_to_check)

    scenario_grouped = detailed_results_df.groupby(["prompt_name", "iteration"])
    scenario_results: Dict[str, Dict[str, Any]] = {}

    def gwet_cols(frame: pd.DataFrame, cols: List[str]) -> float:
        """
        Compute Gwet's AC1/AC2 for specified columns, keeping NaNs.
        Uses irrCAC.raw.CAC(...).gwet().
        """
        data = frame[cols].astype(float)

        # Drop annotators with no data at all (all NaN)
        non_empty_cols = [c for c in cols if not data[c].isna().all()]
        data = data[non_empty_cols]

        if data.shape[1] < 2:
            if verbose:
                print("⚠️ Skipping AC computation: fewer than 2 annotators with valid data.")
            return np.nan

        # Keep only items where at least one rater provided a label
        data = data.loc[~data.isna().all(axis=1)]
        if data.shape[0] < 2:
            if verbose:
                print("⚠️ Skipping AC computation: fewer than 2 items with any annotations.")
            return np.nan

        try:
            cac = CAC(data, weights=ac_weights)
            out = cac.gwet()
            return float(out["est"]["coefficient_value"])
        except Exception as e:
            if verbose:
                print(f"⚠️ Error computing Gwet AC for {cols}: {e}")
            return np.nan

    for (prompt_name, iteration), scenario_group in scenario_grouped:
        scenario_key = f"{prompt_name}_iteration_{iteration}"

        if verbose:
            print(f"\n=== Gwet Panel Difference: {scenario_key} ===")

        run_results: List[Dict[str, Any]] = []

        for run_id in sorted(scenario_group["run"].unique()):
            run_data = scenario_group[scenario_group["run"] == run_id]
            train_data = run_data[run_data["split"] == "train"]
            if len(train_data) == 0:
                continue

            try:
                n_human_annotators = len(annotation_columns)
                if n_human_annotators < 2:
                    if verbose:
                        print(
                            f"  Run {run_id}: Skipping - need at least 2 human annotators, got {n_human_annotators}"
                        )
                    continue

                human_group = annotation_columns

                # (n-1) humans + model (all combinations). If n_human_annotators==2, this yields 2 combos of (1 human + model).
                model_groups = [
                    list(human_subset) + [model_column]
                    for human_subset in itertools.combinations(annotation_columns, n_human_annotators - 1)
                ]

                if verbose:
                    print(f"  Run {run_id}:")
                    print(f"    Using {n_human_annotators} human annotators")
                    print(f"    Model groups: {len(model_groups)} combinations of LLM + {n_human_annotators-1} humans")
                    print(f"    Gwet weights: {ac_weights!r} (identity->AC1, weighted->AC2)")

                ac_H = gwet_cols(train_data, human_group)
                ac_model_groups = [gwet_cols(train_data, group) for group in model_groups]
                ac_M = np.nanmean(ac_model_groups)

                diff = ac_M - ac_H

                run_result = {
                    "run": run_id,
                    "ac_human_groups": ac_H,
                    "ac_model_groups": ac_M,
                    "difference": diff,
                    "human_group": human_group,
                    "model_groups": model_groups,
                    "n_human_annotators": n_human_annotators,
                    "ac_weights": ac_weights,
                }
                run_results.append(run_result)

                if verbose:
                    print(f"    Human panel AC: {ac_H:.4f}" if np.isfinite(ac_H) else "    Human panel AC: nan")
                    print(f"    Model substitution AC: {ac_M:.4f}" if np.isfinite(ac_M) else "    Model substitution AC: nan")
                    print(f"    Δ = model − human = {diff:+.4f}" if np.isfinite(diff) else "    Δ = model − human = nan")

            except Exception as e:
                if verbose:
                    print(f"  Run {run_id}: Error - {str(e)}")

        if run_results:
            mean_ac_H = np.nanmean([r["ac_human_groups"] for r in run_results])
            std_ac_H = np.nanstd([r["ac_human_groups"] for r in run_results])

            mean_ac_M = np.nanmean([r["ac_model_groups"] for r in run_results])
            std_ac_M = np.nanstd([r["ac_model_groups"] for r in run_results])

            mean_diff = np.nanmean([r["difference"] for r in run_results])
            std_diff = np.nanstd([r["difference"] for r in run_results])

            aggregated_metrics = {
                "n_runs": len(run_results),
                "ac_human_panel_mean": mean_ac_H,
                "ac_human_panel_std": std_ac_H,
                "ac_llm_substitution_panel_mean": mean_ac_M,
                "ac_llm_substitution_panel_std": std_ac_M,
                "difference_mean": mean_diff,
                "difference_std": std_diff,
                "ac_weights": ac_weights,
            }

            scenario_results[scenario_key] = {
                "run_results": run_results,
                "aggregated_metrics": aggregated_metrics,
            }

            if verbose:
                print(f"\n  Summary across {len(run_results)} runs:")
                print(f"    Human panel AC: {mean_ac_H:.4f} ± {std_ac_H:.4f}")
                print(f"    Model substitution AC: {mean_ac_M:.4f} ± {std_ac_M:.4f}")
                print(f"    Δ = model − human = {mean_diff:+.4f} ± {std_diff:.4f}")

    return scenario_results


def print_panel_difference_results_gwet(
    results: Dict[str, Dict[str, Any]], show_per_run: bool = False
) -> None:
    for scenario_key, scenario in results.items():
        print(f"\n=== Gwet Panel Difference: {scenario_key} ===")
        agg = scenario["aggregated_metrics"]

        print(f"Human panel AC: {agg['ac_human_panel_mean']:.4f} ± {agg['ac_human_panel_std']:.4f}")
        print(
            f"Model substitution AC: {agg['ac_llm_substitution_panel_mean']:.4f} ± {agg['ac_llm_substitution_panel_std']:.4f}"
        )
        print(f"Δ = model − human = {agg['difference_mean']:+.4f} ± {agg['difference_std']:.4f}")
        print(f"ac_weights: {agg.get('ac_weights')!r} (identity->AC1, weighted->AC2)")

        if show_per_run:
            print("\nDetailed per-run results:")
            for r in scenario["run_results"]:
                print(f"  Run {r['run']}:")
                print(f"    Human panel AC: {r['ac_human_groups']:.4f}" if np.isfinite(r["ac_human_groups"]) else "    Human panel AC: nan")
                print(f"    Model substitution AC: {r['ac_model_groups']:.4f}" if np.isfinite(r["ac_model_groups"]) else "    Model substitution AC: nan")
                print(f"    Δ = model − human = {r['difference']:+.4f}" if np.isfinite(r["difference"]) else "    Δ = model − human = nan")