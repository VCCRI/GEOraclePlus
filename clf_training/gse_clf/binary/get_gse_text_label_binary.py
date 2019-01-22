# Create training data using CREEDS data

import argparse
import os
import sqlite3
import tablib


def main(pert_files, other_files, db_file, output_dir, **kwargs):
    titles = []
    summaries = []

    for files, label in ((pert_files, "pert"), (other_files, "non_pert")):
        new_titles, new_summaries = get_text(db_file, files, label)
        titles += new_titles
        summaries += new_summaries

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    headers = ("text", "label")
    title_outputs = tablib.Dataset(*titles, headers=headers)
    summary_outputs = tablib.Dataset(*summaries, headers=headers)
    prefix = "gse"

    with open("{}/{}_train_binary_title.csv".format(output_dir, prefix), "w") as f:
        f.write(title_outputs.csv)

    with open("{}/{}_train_binary_summary.csv".format(output_dir, prefix), "w") as f:
        f.write(summary_outputs.csv)

    with open("{}/{}_train_binary_all.csv".format(output_dir, prefix), "w") as f:
        f.write(title_outputs.csv)
        summary_outputs.headers = None
        f.write(summary_outputs.csv)


def get_text(db_file, csv_files, label):
    titles = []
    summaries = []
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    for csv_file in csv_files:
        data = tablib.Dataset().load(open(csv_file).read())
        gse_ids = data["geo_id"]
        list_size = 100
        gse_id_chunks = [gse_ids[x:x + list_size] for x in range(0, len(gse_ids), list_size)]

        for gse_id_chunk in gse_id_chunks:
            cur.execute("select title, summary from gse where gse in ({})".format(", ".join("?" for _ in gse_id_chunk)),
                        gse_id_chunk)
            rows = cur.fetchall()

            for row in rows:
                titles.append((row[0], label))
                summaries.append((row[1], label))

    conn.close()

    return titles, summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pert_files",
                        nargs=2,
                        help="The perturbation csv files from CREEDS database")
    parser.add_argument("other_files",
                        nargs=1,
                        help="The non-perturbation csv files from CREEDS database")
    parser.add_argument("db_file",
                        help="The GEO database file")
    parser.add_argument("-o", "--output_dir",
                        default=".",
                        help="Output directory (Default: current directory)")
    parser.set_defaults(method=main)

    args = parser.parse_args()
    args.method(**vars(args))
