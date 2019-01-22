# Get F1 scores from benchmark results

import argparse
import os
import pandas as pd
import re


def main(score_dir, output_file, **kwargs):
    all_scores = pd.DataFrame()
    col_names = None
    index_names = []

    for filename in sorted(os.listdir(score_dir)):
        if not filename.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(score_dir, filename), index_col=False)
        col_names = df["clf"]
        all_scores = all_scores.append(df["f1_scores"])
        data_type = re.search("train_.*_clf", filename).group().replace("train_", "").replace("_clf", "")
        index_names.append(data_type)

    all_scores.index = index_names
    all_scores.index.name = "data_type"
    all_scores.columns = col_names
    all_scores.to_csv(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("score_dir")
    parser.add_argument("output_file")
    parser.set_defaults(method=main)

    args = parser.parse_args()
    args.method(**vars(args))
