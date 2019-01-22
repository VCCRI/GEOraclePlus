# Create training data using CREEDS data

import argparse
import os
import sqlite3
import tablib


def main(csv_files, text_types, db_file, output_dir, **kwargs):
    if len(csv_files) != len(text_types):
        raise argparse.ArgumentError(None, "Number of csv_files has to be identical to the number of text_types")

    title_outputs = tablib.Dataset()
    summary_outputs = tablib.Dataset()
    title_outputs.headers = ("text", "label")
    summary_outputs.headers = ("text", "label")

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    for i, csv_file in enumerate(csv_files):
        data = tablib.Dataset().load(open(csv_file).read())
        gse_ids = data["geo_id"]
        list_size = 100
        gse_id_chunks = [gse_ids[x:x + list_size] for x in range(0, len(gse_ids), list_size)]

        for gse_id_chunk in gse_id_chunks:
            cur.execute("select title, summary from gse where gse in ({})".format(", ".join("?" for _ in gse_id_chunk)),
                        gse_id_chunk)
            rows = cur.fetchall()

            for row in rows:
                title_outputs.append((row[0], text_types[i]))
                summary_outputs.append((row[1], text_types[i]))

    conn.close()

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    with open("{}/pert_train_multi_title.csv".format(output_dir), "w") as f:
        f.write(title_outputs.csv)

    with open("{}/pert_train_multi_summary.csv".format(output_dir), "w") as f:
        f.write(summary_outputs.csv)

    with open("{}/pert_train_multi_all.csv".format(output_dir), "w") as f:
        f.write(title_outputs.csv)
        summary_outputs.headers = None
        f.write(summary_outputs.csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files",
                        nargs=3,
                        help="The csv files from CREEDS database")
    parser.add_argument("text_types",
                        nargs=3,
                        help="The types of text for each csv file")
    parser.add_argument("db_file",
                        help="The GEO database file")
    parser.add_argument("-o", "--output_dir",
                        default=".",
                        help="Output directory (Default: current directory)")
    parser.set_defaults(method=main)

    parser_result = parser.parse_args()
    parser_result.method(**vars(parser_result))
