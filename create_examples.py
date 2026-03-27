import argparse
import pandas as pd
import os


def find_examples(rows_x_raters, cols_x_raters, i):
    """
    Return the subset of rows where all non-NaN ratings in cols_x_raters are equal to i.
    """
    # Mask of cells that are not NaN
    is_not_nan = rows_x_raters[cols_x_raters].notna()

    # Mask of cells equal to i
    is_i = rows_x_raters[cols_x_raters] == i

    # For each row: check if all non-NaN ratings are i
    mask_all_i = (is_i | ~is_not_nan).all(axis=1)

    # Apply mask to select the rows
    dfi = rows_x_raters[mask_all_i]

    return dfi

    
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load a dataset and write examples for the LLM.")
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output examples .txt file")

    args = parser.parse_args()

    input_path = args.input_file
    output_folder = args.output_file
    os.makedirs(output_folder, exist_ok=True) # Ensure output folder exists
    filename = os.path.join(output_folder, "examples_LLM.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("")

    # Load dataset
    df = pd.read_csv(input_path)
    rater_cols = [col for col in df.columns if col.startswith("Rater")]
    print("- Original:", end=" ")
    min_rating = int(df[rater_cols].min().min())
    print(f"from {min_rating}", end=" ")
    max_rating = int(df[rater_cols].max().max())
    print(f"to {max_rating}")

    # Clean to keep only best quality examples
    if "CHAPNUM" in df.columns:
        df = df[~df["full_text"].str.contains(r"\[inaudible\]", case=False, na=False)]

    # Select all columns starting with "Rater"
    rater_cols = [col for col in df.columns if col.startswith("Rater")]
    
    # Compute global min and max across all rating columns
    print("- after cleaning:", end=" ")
    min_rating = int(df[rater_cols].min().min())
    print(f"from {min_rating}", end=" ")
    max_rating = int(df[rater_cols].max().max())
    print(f"to {max_rating}")
    print()

    # Find example within rows annotated by 3 raters
    n_raters = df[rater_cols].notna().sum(axis=1)           # Count how many raters annotated each row (non-NaN)
    rows_3_raters = df[n_raters == 3].copy()                # Keep only rows annotated by exactly 3 raters
    rows_3_raters.dropna(axis=1, how='all', inplace=True)   # Remove column (rater) full of NaN
    cols_3_raters = [c for c in rows_3_raters.columns if c.startswith("Rater")]

    # Find example within rows annotated by 2 raters
    rows_2_raters = df[n_raters == 2].copy()                # Keep only rows annotated by exactly 2 raters
    rows_2_raters.dropna(axis=1, how='all', inplace=True)   # Remove column (rater) full of NaN
    cols_2_raters = [c for c in rows_2_raters.columns if c.startswith("Rater")]
    
    with open(filename, "w", encoding="utf-8") as f:
        for i in range(min_rating, max_rating+1):
            print(f"Rating = {i}")
            dfi = find_examples(rows_3_raters, cols_3_raters, i)
            print(f"- {len(dfi)} rows annotated by 3 raters.")
            
            if dfi.empty:
                # Find example within rows annotated by 2 raters
                dfi = find_examples(rows_2_raters, cols_2_raters, i)
                print(f"- {len(dfi)} rows annotated by 2 raters.")
                if dfi.empty:
                    print(f"No good examples found for rating {i}. Skipping.")
                    continue

            if "CHAPNUM" in dfi.columns and dfi["CHAPNUM"].notna().any():
                # CASE where CHAPNUM exists, then chose the earliest chapter of a lesson
                earliest_chap = dfi.loc[dfi["CHAPNUM"].idxmin()]
                text = earliest_chap["full_text"]
                chap = earliest_chap["CHAPNUM"]
                print(f"Selected chapter: {chap}")
            else:
                # Case where CHAPNUM doesn't exist (whole lesson rating)
                text = dfi["full_text"].sample(1).iloc[0]
            f.write(text)
            f.write(f"\nRating: {i}\n")
            f.write("\n\n")
            print("")

    print(f"Examples text file created")



if __name__ == "__main__":
    main()