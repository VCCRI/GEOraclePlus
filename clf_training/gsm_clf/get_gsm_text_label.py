# Create training data using CREEDS data

import argparse
import os
import sqlite3
import tablib

from collections import defaultdict
from itertools import combinations


def main(csv_files, db_file, output_dir, **kwargs):
    ctrl_ids = []
    pert_ids = []

    for csv_file in csv_files:
        data = tablib.Dataset().load(open(csv_file).read())
        ctrl_ids += get_ids(data, "ctrl_ids")
        pert_ids += get_ids(data, "pert_ids")

    texts = get_text_label((ctrl_ids, pert_ids), ("ctrl", "pert"), db_file)

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    prefix = "gsm"
    all_texts = set()

    for text_type in texts.keys():
        with open("{}/{}_train_{}.csv".format(output_dir, prefix, text_type), "w") as f:
            all_texts.update(texts[text_type])
            output = tablib.Dataset(*texts[text_type], headers=("text", "label"))
            f.write(output.csv)

    with open("{}/{}_train_all.csv".format(output_dir, prefix), "w") as f:
        output = tablib.Dataset(*all_texts, headers=("text", "label"))
        f.write(output.csv)

    for i in range(2, len(texts.keys())):
        for comb in combinations(texts.keys(), i):
            output_texts = set()
            for text_type in comb:
                output_texts.update(texts[text_type])

            with open("{}/{}_train_{}.csv".format(output_dir, prefix, "_".join(sorted(comb))), "w") as f:
                output = tablib.Dataset(*output_texts, headers=("text", "label"))
                f.write(output.csv)


def get_ids(data, id_type):
    all_ids = []
    for ids in data[id_type]:
        all_ids += ids.split("|")

    return all_ids


def get_text_label(id_lists, sample_types, db_file):
    texts = defaultdict(set)
    chunk_size = 100

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    for i, id_list in enumerate(id_lists):
        id_chunks = [id_list[x:x + chunk_size] for x in range(0, len(id_list), chunk_size)]

        for id_chunk in id_chunks:
            cur.execute("select title, description, characteristics_ch1, source_name_ch1 from gsm where gsm in ({})".
                        format(", ".join("?" for _ in id_chunk)), id_chunk)

            for row in cur.fetchall():
                title, description, characteristics, source_name = row
                add_texts(texts, "title", title, sample_types[i])
                add_texts(texts, "desc", description, sample_types[i])
                add_texts(texts, "charcs", characteristics, sample_types[i])
                add_texts(texts, "source", source_name, sample_types[i])

    return texts


def add_texts(texts, text_type, raw_text, sample_type):
    if raw_text:
        texts[text_type].add((raw_text, sample_type))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files",
                        nargs="+",
                        help="The csv files from CREEDS database")
    parser.add_argument("db_file",
                        help="The GEO database file")
    parser.add_argument("-o", "--output_dir",
                        default=".",
                        help="Output directory (Default: current directory)")
    parser.set_defaults(method=main)

    args = parser.parse_args()
    args.method(**vars(args))
