import argparse
import boto3
import h5py
import os
import pandas as pd
import requests
import shlex
import sqlite3
import tempfile

from bs4 import BeautifulSoup
from functools import partial
from pyspark import SparkContext, SparkConf
from subprocess import Popen, PIPE
from urllib.parse import urlparse

MICROARRAY_TYPES = ["in situ oligonucleotide", "oligonucleotide beads", "spotted DNA/cDNA", "spotted oligonucleotide",
                    "mixed spotted oligonucleotide/cDNA"]


def main(input_file, db_file, microarray_script, rna_seq_script, mouse_matrix_file, human_matrix_file, filter_gse,
         strict, pert_agent, min_partitions, region, **kwargs):
    conf = SparkConf().setAppName("GSE Differential Expression Analysis")
    sc = SparkContext(conf=conf)
    results = sc.textFile(input_file, minPartitions=min_partitions)

    gse_gsm_info = results.map(get_gse_gsm_info).filter(lambda x: x is not None)

    if filter_gse:
        gse_gsm_info = gse_gsm_info.filter(partial(filter_by_title_summary, strict))

    if pert_agent:
        gse_gsm_info = gse_gsm_info.filter(partial(filter_by_pert_agent, pert_agent))

    clean_gse_gsm_info = gse_gsm_info.map(get_analysis_info)
    results = clean_gse_gsm_info.map(
        partial(analyse_gse, db_file, microarray_script, rna_seq_script, mouse_matrix_file, human_matrix_file)). \
        filter(lambda x: x is not None)

    final_results = results.reduceByKeyLocally(combine_results)
    keys_filenames = [("microarray", "analysis_results_microarray.csv"), ("rna_seq", "analysis_results_rna_seq.csv"),
                      ("falco", "analysis_falco_info.csv"), ("unknown", "analysis_unknown_tech.csv"),
                      ("error", "analysis_errors.csv")]

    for key, filename in keys_filenames:
        try:
            final_results[key].to_csv(filename)
        except KeyError:
            with open(filename, "w") as f:
                f.write("\n")

    if input_file.startswith("s3://"):
        s3 = boto3.resource("s3", region_name=region)
        o = urlparse(input_file)
        bucket_name = o.netloc

        for _, filename in keys_filenames:
            key = "{}/{}".format(os.path.dirname(o.path.lstrip("/")), filename)
            s3.Bucket(bucket_name).put_object(Key=key, Body=open(filename, "rb"))


def get_gse_gsm_info(line):
    """
    Extract GSE and GSM info
    Args:
        line: the entry to process

    Returns:
        the GSE GSM info tuple
    """
    parts = line.strip().split(",")
    if parts[0] == "gse_id":
        return None

    return parts[0], parts[1:]


def filter_by_title_summary(strict, gse_gsm_info):
    """
    Filter GSE by title and/or summary
    Args:
        strict: boolean to indicate if filter by both title and summary
        gse_gsm_info: the GSE and GSM info tuple

    Returns:
        filtered results
    """
    gse_id, gsm_info = gse_gsm_info
    is_pert_summary, is_pert_title = map(eval, gsm_info[2:4])

    if strict:
        return is_pert_summary and is_pert_title
    else:
        return is_pert_summary or is_pert_title


def filter_by_pert_agent(target_pert_agent, gse_gsm_info):
    """
    Filter GSE by perturbation agent
    Args:
        target_pert_agent: the perturbation agent to be filtered
        gse_gsm_info: the GSE and GSM info tuple

    Returns:
        filtered results
    """
    gse_id, gsm_info = gse_gsm_info
    pert_agent = gsm_info[6].lower()
    target_pert_agent = target_pert_agent.lower()

    return pert_agent in target_pert_agent or target_pert_agent in pert_agent


def get_analysis_info(gse_gsm_info):
    """
    Extract required information for analysis
    Args:
        gse_gsm_info: the GSE and GSM info tuple

    Returns:
        the extracted information tuple
    """
    gse_id, gsm_info = gse_gsm_info
    ctrl_gsm_ids = gsm_info[0].split("|")
    direction = gsm_info[1]
    grouping, organism, pert_agent, pert_gsm_ids = gsm_info[4:]
    pert_gsm_ids = pert_gsm_ids.split("|")

    return gse_id, (organism, direction, pert_agent, ctrl_gsm_ids, pert_gsm_ids, grouping)


def analyse_gse(db_loc, microarray_script, rna_seq_script, mouse_matrix_file, human_matrix_file, gse_gsm_info):
    """
    Perform differential expression analysis on GSEs
    Args:
        db_loc: the location of the database file
        microarray_script: the microarray analysis script location
        rna_seq_script: the RNA-seq analysis script location
        mouse_matrix_file: the mouse matrix file
        human_matrix_file: the human matrix file
        gse_gsm_info: the GSE and GSM info tuple

    Returns:
        the dataframe of differential expression analysis results
    """
    gse_id, gsm_info = gse_gsm_info
    organism, direction, pert_agent, ctrl_gsm_ids, pert_gsm_ids, grouping = gsm_info
    gsm_id = ctrl_gsm_ids[0]

    # Extract technology and organism of GSM
    conn = sqlite3.connect(db_loc)
    c = conn.cursor()
    c.execute("select gpl from gse_gpl where gse = ?", (gse_id,))
    gpl_id = c.fetchone()[0]
    c.execute("select gpl.technology, gsm.organism_ch1 from gpl, gsm "
              "where gsm.gpl = gpl.gpl and gsm.gsm = ?", (gsm_id,))
    tech, organism = c.fetchone()
    conn.close()

    if tech in MICROARRAY_TYPES:
        return analyse_microarray(microarray_script, gse_id, gpl_id, grouping, pert_agent, organism, direction)
    elif tech == "high-throughput sequencing":
        combined_gsm_ids = ctrl_gsm_ids + pert_gsm_ids
        grouping = "0" * len(ctrl_gsm_ids) + "1" * len(pert_gsm_ids)
        organism = organism.lower()

        if organism == "homo sapiens":
            matrix_file = human_matrix_file
        elif organism == "mus musculus":
            matrix_file = mouse_matrix_file
        else:
            matrix_file = None

        # Check if expression values exist in the matrix
        if matrix_file:
            with h5py.File(matrix_file, "r") as f:
                matrix_ids = f["meta"]["Sample_geo_accession"].value

            if all(x.encode("utf-8") in matrix_ids for x in combined_gsm_ids):
                return analyse_rna_seq(rna_seq_script, gse_id, ",".join(combined_gsm_ids), grouping, pert_agent,
                                       organism, direction)

        # RNA-seq data expression values do not exist in the matrix, extract SRA IDs instead
        sra_ids = get_sra_ids(combined_gsm_ids)
        data = {
            "gse_id": gse_id,
            "pert_agent": pert_agent,
            "sra_ids": "|".join(sra_ids),
            "grouping": grouping,
            "organism": organism
        }
        df = pd.DataFrame(data, index=[0])
        df = df.set_index("gse_id")

        return "falco", df

    data = {"gse_id": gse_id, "technology": tech}
    df = pd.DataFrame(data, index=[0])
    df = df.set_index("gse_id")

    return "unknown", df


def analyse_microarray(script, gse_id, gpl_id, grouping, pert_agent, organism, direction):
    """
    Perform differential expression analysis on microarray data
    Args:
        script: the microarray analysis script
        gse_id: the GSE ID
        gpl_id: the GPL ID
        grouping: the microarray grouping info
        pert_agent: the perturbation agent
        organism: the organism
        direction: the perturbation direction

    Returns:
        the dataframe of the analysis results
    """
    with tempfile.NamedTemporaryFile() as tf:
        command = "Rscript {script} {gse_id} {gpl_id} {grouping} {output_file}". \
            format(script=script, gse_id=gse_id, gpl_id=gpl_id, grouping=grouping, output_file=tf.name)
        proc = Popen(shlex.split(command), stdout=PIPE, stderr=PIPE)
        out, err = [x.decode("utf-8") for x in proc.communicate()]

        if proc.returncode != 0 or "[Errno" in err:
            if "404 Not Found" in err:
                return "microarray", pd.DataFrame()

            data = {"gse_id": gse_id, "command": command}
            df = pd.DataFrame(data, index=[0])
            df = df.set_index("gse_id")

            return "error", df

        df = pd.read_csv(tf.name, sep="\t")
        df["gse_id"] = gse_id
        df["pert_agent"] = pert_agent
        df["organism"] = organism
        df["direction"] = direction
        df = df.set_index("gse_id")

    return "microarray", df


def analyse_rna_seq(script, gse_id, gsm_ids, grouping, pert_agent, organism, direction):
    """
    Perform differential expression analysis on RNA-seq data
    Args:
        script: the RNA-seq analysis script
        gse_id: the GSE ID
        gsm_ids: the GSM IDs
        grouping: the RNA-seq grouping info
        pert_agent: the perturbation agent
        organism: the organism
        direction: the perturbation direction

    Returns:
        the dataframe of the analaysis results
    """
    with tempfile.NamedTemporaryFile() as tf:
        grouping = ",".join(grouping)
        command = "Rscript {script} {gsm_ids} {grouping} {output_file}". \
            format(script=script, gsm_ids=gsm_ids, grouping=grouping, pert_agent=pert_agent, gse_id=gse_id,
                   output_file=tf.name)
        proc = Popen(shlex.split(command), stdout=PIPE, stderr=PIPE)
        out, err = [x.decode("utf-8") for x in proc.communicate()]

        if proc.returncode != 0 or "[Errno" in err:
            data = {"gse_id": gse_id, "command": command}
            df = pd.DataFrame(data, index=[0])
            df = df.set_index("gse_id")

            return "error", df

        df = pd.read_csv(tf.name, sep="\t")
        df["gse_id"] = gse_id
        df["pert_agent"] = pert_agent
        df["organism"] = organism
        df["direction"] = direction
        df = df.set_index("gse_id")

    return "rna_seq", df


def get_sra_ids(gsm_ids):
    """
    Get SRA IDs
    Args:
        gsm_ids: the GSM IDs

    Returns:
        the list of SRA IDs
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=sra&term={}".format(" or ".join(gsm_ids))
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "xml")
    uid_list = [x.text for x in soup.find_all("Id")][::-1]

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&rettype=runinfo&id={}". \
        format(",".join(uid_list))
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "xml")

    return [x.text for x in soup.find_all("Run")]


def combine_results(df1, df2):
    return df1.append(df2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-i", "--input_file", required=True, help="The classified results file")
    required_args.add_argument("-d", "--db_file", required=True, help="The GEOmetadb file")
    required_args.add_argument("-ms", "--microarray_script", required=True,
                               help="The location of the microarray analysis script")
    required_args.add_argument("-rs", "--rna_seq_script", required=True,
                               help="The location of the microarray analysis script")
    required_args.add_argument("-mm", "--mouse_matrix_file", required=True, help="The mouse matrix file")
    required_args.add_argument("-hm", "--human_matrix_file", required=True, help="The human matrix file")

    parser.add_argument("-f", "--filter_gse", action="store_true",
                        help="If set to true, GSE will be fitered based on the classification result on the title and "
                             "summary")
    parser.add_argument("-s", "--strict", action="store_true",
                        help="If set to true, both title and summary of GSE must be classified as a perturbation text "
                             "for further processing")
    parser.add_argument("-a", "--pert_agent", help="The expected perturbation agent")
    parser.add_argument("-p", "--min_partitions", type=int, default=2, help="The minimum number of partitions")
    parser.add_argument("-r", "--region", default="us-west-2", help="AWS region (Default: %(default)s)")
    parser.set_defaults(method=main)

    args = parser.parse_args()
    args.method(**vars(args))
