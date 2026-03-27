"""
utils.py

This module provides utility functions for computing metrics.

Functions:
    - compute_human_accuracies(df, annotation_columns, ground_truth_column="GroundTruth"):
      Computes the accuracy for each human annotator.

    - compute_majority_vote(annotations, ignore_na=True):
      Computes the majority vote for each instance across multiple annotators.
      
    - ensure_numeric_columns(df, columns, nullable=True):
      Ensures that specified columns in a DataFrame are converted to numeric types.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List


def compute_human_accuracies(df, annotation_columns, ground_truth_column="GroundTruth"):
    """
    Computes the accuracy for each human annotator using sklearn's accuracy_score.

    Parameters:
      df (pd.DataFrame): DataFrame containing the annotator columns and ground truth.
      annotation_columns (List[str]): List of human annotator column names.
      ground_truth_column (str): Name of the column with the ground truth labels.

    Returns:
      Dict[str, float]: A dictionary mapping each annotator to their accuracy.
    """
    from sklearn.metrics import accuracy_score

    accuracies = {}
    for col in annotation_columns:
        # Filter out invalid annotations.
        valid_mask = df[col].notnull() & (df[col] != "")
        if valid_mask.sum() > 0:
            y_true = df.loc[valid_mask, ground_truth_column]
            y_pred = df.loc[valid_mask, col]
            accuracies[col] = accuracy_score(y_true, y_pred)
        else:
            accuracies[col] = float("nan")
    return accuracies


def compute_majority_vote(annotations: Dict[str, List], ignore_na: bool = True) -> List:
    """
    Computes the majority vote for each instance across multiple annotators.

    Parameters:
    ----------
    annotations : Dict[str, List]
        Dictionary where keys are annotator names and values are lists of annotations.
    ignore_na : bool, optional
        If True, ignores NA values when computing the majority vote. Default is True.

    Returns:
    -------
    List
        A list containing the majority vote for each instance.
    """
    if not annotations:
        return []

    # Get the number of instances
    n_instances = len(next(iter(annotations.values())))

    # Check that all annotators have the same number of instances
    for annotator, labels in annotations.items():
        if len(labels) != n_instances:
            raise ValueError(
                f"Annotator {annotator} has {len(labels)} instances, expected {n_instances}"
            )

    # Compute majority vote for each instance
    majority_votes = []
    for i in range(n_instances):
        instance_labels = []
        for annotator, labels in annotations.items():
            label = labels[i]
            if ignore_na and (pd.isna(label) or label == ""):
                continue
            instance_labels.append(label)

        if not instance_labels:
            # If all annotations are NA, use NA as the majority vote
            majority_votes.append(np.nan)
        else:
            # Count occurrences of each label
            label_counts = Counter(instance_labels)
            # Find the label with the highest count
            majority_label, _ = label_counts.most_common(1)[0]
            majority_votes.append(majority_label)

    return majority_votes


def ensure_numeric_columns(
    df: pd.DataFrame, columns: List[str], nullable: bool = True
) -> pd.DataFrame:
    """
    Ensures that specified columns in a DataFrame are converted to numeric types.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing the columns to convert.
    columns : List[str]
        List of column names to convert to numeric types.
    nullable : bool, optional
        If True, uses pandas nullable integer type (Int64) which can handle NaN values.
        If False, uses standard integer type (int64) which cannot handle NaN values.
        Default is True.

    Returns:
    -------
    pd.DataFrame
        A copy of the input DataFrame with specified columns converted to numeric types.
    """
    df_copy = df.copy()

    for col in columns:
        if col in df_copy.columns:
            # Convert to numeric, coercing errors to NaN
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

            # Convert to appropriate integer type
            if nullable:
                df_copy[col] = df_copy[col].astype("Int64")  # Nullable integer type
            else:
                # Fill NaN values before converting to non-nullable type
                df_copy[col] = df_copy[col].fillna(-1).astype("int64")

    return df_copy
