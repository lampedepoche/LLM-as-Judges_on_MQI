import pandas as pd
import itertools
import numpy as np
import krippendorff
from pathlib import Path
import os

# *** Download and read the files ***

def get_base_dir() -> Path:
    """
    Return the base directory of the current script.  
    If executed from a notebook (where __file__ is undefined), returns the current working directory instead.
    """
    try:
        return Path(__file__).resolve().parent # get the path to the python file
    except NameError:
        # In a notebook, __file__ doesn't exist
        return Path.cwd()
    
def get_path(filename):
    """
    Return the file paths for the dataset.  
    """
    base = get_base_dir().parent

    return base / "datasets" / filename

def read_files(filename):
    """
    Load the dataset into pandas DataFrames.  
    Handles data types and encoding for consistent downstream processing.
    """
    DATASET_PATH = get_path(filename)

    df = pd.read_csv(
        DATASET_PATH,
        dtype={"NCTETID":"string", "OBSID": "string", 'DISTRICT': "string", "SCHOOLYEAR_SP": "string"},
        encoding="utf-8",
        low_memory=False,
    )
    return df


def infer_level(df,rater_cols):
    # Infer rating value domain across all raters
    ratings = df[rater_cols].to_numpy().ravel()
    ratings = ratings[~pd.isna(ratings)]
    unique_vals = np.unique(ratings)

    # Choose level: nominal if binary, otherwise ordinal
    if len(unique_vals) == 2:
        return "nominal"
    else:
        return "ordinal"

# Compute global Krippendorff alpha
def compute_alpha_for_indices(indices, df, rater_cols, level):
    subset = df.loc[sorted(indices)]
    data = subset[rater_cols].to_numpy().T
    try:
        alpha = krippendorff.alpha(
            reliability_data=data,
            level_of_measurement=level
        )
    except ValueError as e:
        msg = str(e).lower()
        # Specific case where all ratings have a single value, consider as no agreement
        if "one value in the domain" in msg:
            alpha = 0
        else:
            raise e
    return alpha


def create_best_subsample(filename: str, target_n: int = 50, min_alpha: float = 0.70):
    """
    Build a high-reliability subsample based on pairwise Krippendorff's alpha.

    Steps:
    1. Compute pairwise alpha for all rater pairs.
    2. Select the best pairs until at least `target_n` unique items are included.
    3. Then continue adding pairs as long as the global alpha on the growing subset
       stays >= `min_alpha`.
    4. Return the original dataframe filtered to the selected items,
       the final global alpha, and the list of selected pairs.
    """
    df = read_files(filename)

    # 1. Identify rater columns and infer value domain
    rater_cols = [c for c in df.columns if c.startswith("Rater ")]
    level=infer_level(df, rater_cols)

    # 2. Matrices for pairwise alpha and number of common items
    result = pd.DataFrame(index=rater_cols, columns=rater_cols, dtype=float)
    n_common = pd.DataFrame(index=rater_cols, columns=rater_cols, dtype=float)

    # 3. Compute pairwise Krippendorff alpha
    for r1, r2 in itertools.combinations(rater_cols, 2):
        sub = df[[r1, r2]].dropna()
        if len(sub) < 10:
            continue

        data = np.array([sub[r1].values, sub[r2].values])
        try:
            alpha = krippendorff.alpha(
                reliability_data=data,
                level_of_measurement=level
            )
        except ValueError as e:
            msg = str(e).lower()
            # Specific case where all ratings have a single value, consider as perfect no agreement
            if "one value in the domain" in msg:
                alpha = 0
            else:
                raise e

        result.loc[r1, r2] = alpha
        n_common.loc[r1, r2] = len(sub)

    # 4. Sort pairs by highest alpha (and highest n_common as tie-breaker)
    upper_mask = np.triu(np.ones(result.shape, dtype=bool), k=1)
    pairs_alpha = result.where(upper_mask).stack()
    pairs_n     = n_common.where(upper_mask).stack()

    pairs = (
        pd.DataFrame({"alpha": pairs_alpha, "n_common": pairs_n})
        .dropna()
        .sort_values(by=["alpha", "n_common"], ascending=[False, False])
    )

    selected_pairs = []
    selected_indices = set()

    # 5. First phase: reach at least target_n items
    for (r1, r2), row in pairs.iterrows():
        sub = df[[r1, r2]].dropna()
        idx = sub.index
        if len(idx) == 0:
            continue

        # Add full pair of indices
        selected_indices.update(idx.tolist())
        selected_pairs.append((r1, r2))

        if len(selected_indices) >= target_n:
            break

    # Compute alpha for the initial subset
    current_alpha = compute_alpha_for_indices(selected_indices, df, rater_cols, level)

    # 6. Second phase: keep adding pairs while global alpha stays >= min_alpha
    for (r1, r2), row in pairs.iterrows():
        if (r1, r2) in selected_pairs:
            continue

        sub = df[[r1, r2]].dropna()
        idx = sub.index
        if len(idx) == 0:
            continue

        # Try adding this pair
        new_indices = set(selected_indices)
        new_indices.update(idx.tolist())

        new_alpha = compute_alpha_for_indices(new_indices, df, rater_cols, level)

        # Accept the pair only if alpha stays above the threshold
        if new_alpha >= min_alpha:
            selected_indices = new_indices
            selected_pairs.append((r1, r2))
            current_alpha = new_alpha
        else:
            # Stop as soon as alpha would drop below the threshold
            break

    # 7. Final subset of the original dataframe
    selected_indices = sorted(set(selected_indices)) # Remove duplicates (case of sample rated by 3 raters)
    final_df = df.loc[sorted(selected_indices)].copy()
    final_df = final_df.dropna(axis=1, how="all") # Remove raters that are not needed in this subset

    # Compute number of pairs and number of raters
    n_pairs = len(selected_pairs)
    unique_raters = sorted({r for pair in selected_pairs for r in pair})
    n_raters = len(unique_raters)

    print("\n=== Subsample Summary ===")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Final Krippendorff's alpha: {current_alpha:.3f}")
    print(f"Number of selected pairs: {n_pairs}")
    print(f"Number of unique raters involved: {n_raters}")

    return final_df, current_alpha, selected_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a high-agreement subsample based on inter-rater reliability."
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Input dataset filename"
    )
    args = parser.parse_args()

    final_df, final_alpha, selected_pairs = create_best_subsample(filename=args.filename)
    print('done')

