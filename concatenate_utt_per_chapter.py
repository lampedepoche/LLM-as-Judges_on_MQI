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
    
def get_paths():
    """
    Return the file paths for the annotation and utterance datasets.  
    """
    base = get_base_dir()
    mqi = base / "ICPSR_36095" / "2 Mathematical Quality of Instruction" / "MQI_Data.tsv"
    clas = base / "ICPSR_36095" / "1 Class observations" / "36095-0001-Data.tsv"
    utt = base / "ncte_single_utterances.csv"

    return mqi, clas, utt

def read_files():
    """
    Load the annotation and utterance datasets into pandas DataFrames.  
    Handles data types, encoding, and missing values for consistent downstream processing.
    """
    MQI_PATH, CLASS_PATH, UTTERANCES_PATH = get_paths()

    mqi_df = pd.read_csv(
        MQI_PATH,
        sep="\t",
        dtype={"OBSID": "string", 'CHAPNUM': "Int64", "SEGMENT": "Int64"},
        na_values={"CHAPNUM": [" "]},
        encoding="utf-8",
        low_memory=False,
    )
    class_df = pd.read_csv(
        CLASS_PATH,
        sep="\t",
        dtype={"OBSID": "string", 'CHAPNUM': "Int64"},
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
    return mqi_df, class_df, utterances_df


# *** First cleaning and subsetting the dataframes ***

def cleaning(mqi_df: pd.DataFrame, class_df: pd.DataFrame, utterances_df: pd.DataFrame):
    """
    Clean the annotation DataFrame by removing problematic lessons, filtering out
    invalid annotation values, keeping only relevant columns, and grouping by chapters.
    """
    bad_obsids = ["4263", "2065"]
    cols_ann = ["OBSID", "CHAPNUM"]

    # mqi_df
    mqi_lesson_df = (
        mqi_df
        .loc[(~mqi_df['OBSID'].isin(bad_obsids)) & # Remove the lessons with issues
            (mqi_df["SEGMENT"] == 1),              # Keep only annotations made on chapter
            cols_ann                               # Only keep the interesting columns 
        ]
        .groupby('OBSID')['CHAPNUM']    # group by lesson
        .max()          # keep the max chapter number (to avoid issues with chapters non-sequentially numbered)
        .reset_index()
        .rename(columns={'CHAPNUM': 'N_CHAP'})
        .copy()
    )

    # class_df
    class_lesson_df = (
        class_df
        .loc[(~class_df['OBSID'].isin(bad_obsids)), # Remove the lessons with issues
            cols_ann                               # Only keep the interesting columns 
        ]
        .groupby('OBSID')['CHAPNUM']    # group by lesson
        .max()          # keep the max chapter number (to avoid issues with chapters non-sequentially numbered)
        .reset_index()
        .rename(columns={'CHAPNUM': 'N_CHAP'})
        .copy()
    )

    # Utterances_df
    cols_utt = ["OBSID", "combined_txt", "num_words", "turn_idx"]
    subset_utt_df = (
        utterances_df
        .assign(
            combined_txt=lambda d: d['speaker'].fillna('___').astype(str) # concatenate the speaker with their text
                .str.cat(d['text'].fillna('___').astype(str), sep=': ')
        )
        .loc[~utterances_df['OBSID'].isin(bad_obsids), cols_utt] # Only keep the interesting columns + remove lessons with issues
        .copy()
    )
    return mqi_lesson_df, class_lesson_df, subset_utt_df


# *** Words per chapter assignment ***

def build_windows(total_words: int, n_chap: int) -> pd.DataFrame:
    """
    Create sequential chapter windows.
    Windows cover the entire word range and are sized as equally as possible.
    No utterance is ever cut because assignment is based on utterance midpoints.
    """
    n_chap = max(int(n_chap), 1)
    total_words = max(int(total_words), 0)

    # Edge case: no words in the lesson
    if total_words == 0:
        return pd.DataFrame(
            [(1, 0, 0)],
            columns=["CHAPNUM", "chap_start", "chap_end"]
        )

    # Compute base window length and how many windows receive one extra word
    base_len, remainder = divmod(total_words, n_chap)

    windows = []
    start = 1

    for c in range(1, n_chap + 1):
        # First `remainder` chapters get one extra word
        length = base_len + (1 if c <= remainder else 0)
        end = start + length - 1

        windows.append((c, start, end))
        start = end + 1

        if start > total_words:
            break

    return pd.DataFrame(windows, columns=["CHAPNUM", "chap_start", "chap_end"])



def assign_chapters_per_obsid(g: pd.DataFrame, n_chap: int):
    """
    Assign a chapter number to each utterance within a lesson based on its word-span midpoint.  
    Uses precomputed chapter windows to approximate which segment each utterance belongs to.
    """
    total_words = int(g["end_word"].max()) if len(g) else 0
    win = build_windows(total_words, n_chap)

    # midpoint of each utterance span
    midpoints = ((g["start_word"] + g["end_word"]) // 2).to_numpy()
    starts = win["chap_start"].to_numpy()
    ends = win["chap_end"].to_numpy()

    chapnum = np.empty(len(midpoints), dtype=int)
    for i, m in enumerate(midpoints):
        mask = (starts <= m) & (m <= ends)
        chapnum[i] = win.loc[mask, "CHAPNUM"].iloc[0] if mask.any() else win["CHAPNUM"].iloc[-1]

    g = g.assign(CHAPNUM=chapnum)
    return g


def assign_chapters(utter_df: pd.DataFrame, lessons_df: pd.DataFrame):
    """
    Assign chapter numbers to all utterances across lessons based on word spans.  
    Computes cumulative word positions per lesson, builds chapter windows,  
    and labels each utterance with its corresponding approximate chapter.
    """
    # Map OBSID to number of chapters
    nchap_map = lessons_df.set_index("OBSID")["N_CHAP"].astype(int).to_dict()

    # Compute spans per lesson
    utter_spans = utter_df.sort_values(['OBSID', 'turn_idx']).copy()
    w = utter_spans['num_words'].fillna(0).astype(int)
    utter_spans['end_word']   = w.groupby(utter_spans['OBSID']).cumsum()
    utter_spans['start_word'] = utter_spans['end_word'] - w + 1

    # Assign chapters per lesson
    out_groups = []
    for obsid, g in utter_spans.groupby("OBSID"):
        n_chap = int(nchap_map.get(obsid, 1))
        g2 = assign_chapters_per_obsid(g, n_chap=n_chap)
        out_groups.append(g2)

    utter_with_chap = pd.concat(out_groups, ignore_index=True) if out_groups else utter_spans
    return utter_with_chap



# *** Group utterances by unit of annotation (chapter or lesson) ***

def group_utterances_by_chapter(utter_with_chap: pd.DataFrame) -> pd.DataFrame:
    """
    Group all utterances within each chapter and concatenate their text in order.  
    Produces a single text block per chapter representing the full spoken content of that segment.
    """
    all_utt_chap = (
        utter_with_chap
        .sort_values(['OBSID', 'CHAPNUM', 'turn_idx'])
        .groupby(['OBSID', 'CHAPNUM'], as_index=False)
        .agg(full_text=('combined_txt', lambda s: '\n'.join(s.fillna('').astype(str))))
    )
    return all_utt_chap

def group_utterances_by_lesson(utter_with_chap: pd.DataFrame) -> pd.DataFrame:
    """
    Group all utterances within each lesson and concatenate their text in order.  
    Produces a single text block per lesson representing the full spoken content of that lesson.
    """
    all_utt_lesson = (
        utter_with_chap
        .sort_values(['OBSID', 'turn_idx'])
        .groupby(['OBSID'], as_index=False)
        .agg(full_text=('combined_txt', lambda s: '\n'.join(s.fillna('').astype(str))))
    )
    return all_utt_lesson


# *** Pipeline execution *** #
if __name__ == "__main__":
    # Download and clean datasets
    MQI_annotation_df, CLASS_annotation_df, utterances_df = read_files()
    MQI_lesson_df, CLASS_lesson_df, subset_utt_df = cleaning(MQI_annotation_df, CLASS_annotation_df, utterances_df)

    # Add chapter assignments to utterances
    MQI_utter_with_chap = assign_chapters(subset_utt_df, MQI_lesson_df)
    CLASS_utter_with_chap = assign_chapters(subset_utt_df, CLASS_lesson_df)

    # dataset with utterances concatenated per chapter
    MQI_all_utt_lesson = group_utterances_by_lesson(MQI_utter_with_chap)

    # datasets with utterances concatenated per chapter
    MQI_all_utt_chap = group_utterances_by_chapter(MQI_utter_with_chap)
    CLASS_all_utt_chap = group_utterances_by_chapter(CLASS_utter_with_chap)

    # CSV exports
    MQI_all_utt_lesson.to_csv("MQI_all_utt_lesson.csv", index=False)
    MQI_all_utt_chap.to_csv("MQI_all_utt_chap.csv", index=False)
    CLASS_all_utt_chap.to_csv("CLASS_all_utt_chap.csv", index=False)