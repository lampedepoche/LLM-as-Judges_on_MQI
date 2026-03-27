import pandas as pd
import numpy as np
from pathlib import Path



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
    
def get_paths(lesson_level: bool, MQI: bool = True):
    """
    Return the file paths for the annotation and utterance datasets.  
    Selects the MQI dataset by default, or the CLASS Observations dataset if specified.
    """
    base = get_base_dir()
    if MQI: # Use the Mathematical Quality of Instruction (MQI) dataset
        ann = base / "ICPSR_36095" / "2 Mathematical Quality of Instruction" / "MQI_Data.tsv"
        if lesson_level:
            utt = base / "Segmented transcripts" / "MQI_all_utt_lesson.csv"   # Use utterances grouped by entire lesson (OBSID)
        else:
            utt = base / "Segmented transcripts" / "MQI_all_utt_chap.csv"     # Use utterances grouped by chapter (7.5 min)
    else:
        ann = base / "ICPSR_36095" / "1 Class observations" / "36095-0001-Data.tsv"
        utt = base / "Segmented transcripts" / "CLASS_all_utt_chap.csv"       # Use utterances grouped by chapter (15 min)
    return ann, utt

def read_files(lesson_level: bool, MQI: bool = True):
    """
    Load the annotation and utterance datasets into pandas DataFrames.  
    Handles data types, encoding, and missing values for consistent downstream processing.
    """
    ANNOTATION_PATH, UTTERANCES_PATH = get_paths(lesson_level, MQI)

    annotation_df = pd.read_csv(
        ANNOTATION_PATH,
        sep="\t",
        dtype={"OBSID": "string", "RATERID": "string", 'NCTETID': "string", 'CHAPNUM': "Int64", 'DISTRICT': "string", 'SCHOOLYEAR_SP': "string"},
        na_values={"CHAPNUM": [" "]},
        encoding="utf-8",
        low_memory=False,
    )
    utterances_df = pd.read_csv(
        UTTERANCES_PATH,
        dtype={"OBSID": "string", "comb_idx": "string"},
        encoding="utf-8",
        low_memory=False,
    )
    return annotation_df, utterances_df



# *** First cleaning and subsetting the dataframes ***

def cleaning(ann_df: pd.DataFrame, col: str):
    """
    Keeping only relevant columns.
    """
    # ann_df
    if col not in ann_df.columns:
        raise KeyError(f"Column '{col}' not found in annotations.")
    
    cols_ann = ["NCTETID", "OBSID", "RATERID", "CHAPNUM", "DISTRICT", "SCHOOLYEAR_SP", col]

    return ann_df[cols_ann].copy()



# *** Pivoting the dataframes ***

def _rename_rater_columns(pivoted_df, ann_df):
    """
    Rename columns corresponding to rater IDs by prefixing them with 'Rater'.  
    """
    raters = set(ann_df["RATERID"].astype(str).unique().tolist())
    new_cols = []
    for c in pivoted_df.columns:
        sc = str(c)
        new_cols.append(f"Rater {sc}" if sc in raters else c)
    pivoted_df.columns = new_cols
    return pivoted_df


def pivot_annotation(ann_df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Reshape the annotation DataFrame to have one row per entity (chapter or lesson) and one column per rater.  
    Each cell contains the selected annotation value for a given rater.
    """
    # Keep all columns except RATERID and the selected column as index
    index_cols = [c for c in ann_df.columns if c not in ('RATERID', col)]
    
    pivoted_df = (
        ann_df
        .pivot_table(
            index=index_cols,
            columns='RATERID',
            values=col,
            aggfunc='first' # In case of duplicates, take the first occurrence
        )
        .reset_index()
    )
    return _rename_rater_columns(pivoted_df, ann_df)


# *** Merge the dataframes ***

def merge_datasets_CHAPNUM(ann_df: pd.DataFrame, utter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the utterance DataFrame with the annotation DataFrame on CHAPNUM and OBSID.
    """
    merged_df = pd.merge(
        left=ann_df,
        right=utter_df,
        on=['OBSID', 'CHAPNUM'],
        how='inner' # only keep common data
    )
    merged_df = merged_df[["NCTETID"] + [c for c in merged_df.columns if c != "NCTETID"]]
    return merged_df

def merge_datasets_OBSID(ann_df: pd.DataFrame, utter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the utterance DataFrame with the annotation DataFrame on OBSID.
    """
    merged_df = pd.merge(
        left=ann_df,
        right=utter_df,
        on=['OBSID'],
        how='inner' # only keep common data
    )
    merged_df = merged_df[["NCTETID"] + [c for c in merged_df.columns if c != "NCTETID"]]
    return merged_df



# *** Main function to prepare the dataset ***

def prepare_dataset(col: str, MQI: bool = True):
    """
    End-to-end pipeline to produce the analysis-ready dataset for a given annotation column.
    Loads raw files, cleans annotations and utterances, pivots rater scores, estimates chapter
    windows from word counts (≈7.5-minute segments), assigns chapters to utterances, and merges
    texts with rater annotations at chapter or lesson level depending on the column.
    """
    lesson_cols = [
    "ORIENT", "SUMM", "MQI_CHECK", "DIFFINST", "LLC", "MQI3", "MKT3", "MQI5", "MKT5",
    "TSTUDEA", "TREMSTU", "STUENG", "CLMATINQ", "LESSEFFIC", "DENSMAT",
    "LATASK", "LESSCLEAR", "TASKDEVMAT", "ERRANN", "WORLD"
    ]
    lesson_level = col in lesson_cols

    # Download the data
    annotation_df, utterances_df = read_files(lesson_level, MQI=MQI)

    # Clean the data
    clean_ann_df = cleaning(annotation_df, col)
    # If the selected column is in lesson-level columns, no need to keep CHAPNUM
    if lesson_level:
        clean_ann_df.drop(columns=['CHAPNUM'], inplace=True)
    # If MQI dataset, filter out invalid annotation values (999)
    if MQI:
        clean_ann_df = clean_ann_df.loc[clean_ann_df[col] != 999]

    # Pivot the annotation dataframe
    pivot_ann_df = pivot_annotation(clean_ann_df, col)

    # Merge according to chapter-level or lesson-level annotations
    if lesson_level:
        df = merge_datasets_OBSID(pivot_ann_df, utterances_df)
    else:
        df = merge_datasets_CHAPNUM(pivot_ann_df, utterances_df)

    # Convert rater columns to integer type and handle missing values (998 as NA)
    rater_cols = [c for c in df.columns if str(c).startswith("Rater ")]
    df[rater_cols] = df[rater_cols].astype("Int64")
    df.loc[:, rater_cols] = df[rater_cols].replace(998, pd.NA)
    df = df.dropna(axis=1, how='all') # remove columns full of NaN

    return df

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Load NCTE datasets")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--mqi", dest="MQI", action="store_true", help="Use MQI dataset")
    g.add_argument("--class_obs", dest="MQI", action="store_false", help="Use CLASS Observations dataset")

    p.add_argument("--col", type=str, required=True, help="Column to process")
    p.set_defaults(MQI=True)

    args = p.parse_args()

    df = prepare_dataset(col=args.col, MQI=args.MQI)
    print(f"df file: {df.shape}")