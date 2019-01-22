import argparse
import boto3
import jellyfish
import numpy as np
import os
import pandas as pd
import pickle
import re
import shlex
import sqlite3
import tablib
import tempfile

from collections import defaultdict
from functools import partial
from mygene import MyGeneInfo
from pyspark import SparkContext, SparkConf
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from subprocess import Popen, PIPE
from urllib.parse import urlparse

MIN_EPS = 0.1
MAX_EPS = 0.9
SPECIAL_CHARS = ['!', '#', '$', '&', "'", '(', ')', '*', '+', ',', '/', ':', ';', '=', '?', '@', '[', ']']
UP_FEATS = ["overexp", "express", "transgen", "expos", "tg", "induc"]
DOWN_FEATS = ["knock", "null", "ko", "s[hi]rna", "delet", "reduc", "\\-\\/", "\\/\\-", "\\+\\/", "\\/\\+", "cre",
              "flox", "mut", "defici"]


def main(input_file, db_file, gse_title_pickle, gse_summary_pickle, gsm_pickle, noble_coder, and_keywords, or_keywords,
         organism, upload_loc, min_partitions, region, **kwargs):
    with tempfile.NamedTemporaryFile() as tf, tempfile.TemporaryDirectory() as dirname:
        command = "java -jar {} -terminology NCI_Thesaurus -input {} -output {}".format(noble_coder, tf.name, dirname)
        proc = Popen(shlex.split(command), stdout=PIPE, stderr=PIPE)
        proc.communicate()

    gse_clf_title = pickle.load(open(gse_title_pickle, "rb"))
    gse_clf_summary = pickle.load(open(gse_summary_pickle, "rb"))
    gsm_clf = pickle.load(open(gsm_pickle, "rb"))
    conf = SparkConf().setAppName("GSE Classification")
    sc = SparkContext(conf=conf)

    if input_file is not None:
        gse_ids = sc.textFile(input_file, minPartitions=min_partitions)
    else:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute("select gse from gse")
        gse_ids = sc.parallelize(c.fetchall(), numSlices=min_partitions)
        conn.close()

    # Classify GSE title and summary
    gse_pert = gse_ids.map(partial(classify_gse_title_summary, db_file, organism, gse_clf_title, gse_clf_summary)). \
        filter(lambda x: x is not None)

    # Extract GSE title, GSM IDs, GSM texts and classify sample type
    gse_gsm_info = gse_ids.map(partial(get_gsm_info, db_file, organism, and_keywords, or_keywords, gsm_clf))

    # Check if there are both perturbation and control GSM samples in a GSE
    valid_gse_gsm_info = gse_gsm_info.filter(filter_ctrl_pert)

    # Cluster GSM samples
    gse_gsm_clusters = valid_gse_gsm_info.flatMap(partial(cluster_gsm, noble_coder))

    # Combine clustering results back under the same GSE
    gse_all_clusters = gse_gsm_clusters.aggregateByKey(([], []), merge_two_tuples, merge_two_lists)

    # Final filtering to match the best perturbation samples to control samples
    gse_final_clusters = gse_all_clusters.map(get_target_pert_indices).filter(lambda x: x is not None)

    # Tidy up and save results
    results_file = "gse_classified_results.csv"
    try:
        results = gse_final_clusters.join(gse_pert).map(process_results).reduce(combine_results)
        results.to_csv(results_file)
    except ValueError:
        with open(results_file, "w") as f:
            f.write("\n")

    if input_file.startswith("s3://") or upload_loc is not None:
        if input_file is not None:
            o = urlparse(input_file)
        else:
            o = urlparse(upload_loc)

        bucket_name = o.netloc
        key = "{}/{}".format(os.path.dirname(o.path.lstrip("/")), results_file)
        s3 = boto3.resource("s3", region_name=region)
        s3.Bucket(bucket_name).put_object(Key=key, Body=open(results_file, "rb"))


def classify_gse_title_summary(db_loc, organism, clf_title, clf_summary, gse_id):
    """
    Classify if the GSE title and summary indicates the GSE is a perturbation experiment
    Args:
        db_loc: the location of the database file
        organism: the organism
        clf_title: the title classifier
        clf_summary: the summary classifier
        gse_id: the GSE ID

    Returns:
        key: the GSE ID
        value: a list of booleans
    """
    query = "select gse.title, gse.summary, gse.overall_design, gsm.organism_ch1 from gse, gsm, gse_gsm " \
            "where gse.gse = ? and gse_gsm.gse = gse.gse and gse_gsm.gsm = gsm.gsm"
    if organism:
        query += " and lower(gsm.organism_ch1) = \"{}\"".format(organism.lower())

    conn = sqlite3.connect(db_loc)
    c = conn.cursor()
    c.execute(query, (gse_id,))
    row = c.fetchone()
    conn.close()

    if row:
        title, summary, design, organism = row
        text = title + " " + summary

        if design is not None:
            text += " " + design

        up_count = sum(len(re.findall(x, text, flags=re.IGNORECASE)) for x in UP_FEATS)
        down_count = sum(len(re.findall(x, text, flags=re.IGNORECASE)) for x in DOWN_FEATS)

        if up_count > down_count:
            direction = "+"
        else:
            direction = "-"

        title_result = clf_title.predict([title])[0]
        summary_result = clf_summary.predict([summary])[0]

        return gse_id, [title_result == "pert", summary_result == "pert", organism, direction]
    else:
        return None


def get_gsm_info(db_loc, organism, and_keywords, or_keywords, clf, gse_id):
    """
    Extract GSE title, GSM IDs, GSM texts and classify sample type
    Args:
        db_loc: the location of the database file
        organism: the organism
        and_keywords: the list of and keywords
        or_keywords: the list of or keywords
        clf: the sample type classifier
        gse_id: the GSE ID

    Returns:
        key: the GSE ID
        value: GSE title, GSM IDs, GSM texts, GSM classified types
    """
    conn = sqlite3.connect(db_loc)
    c = conn.cursor()
    c.execute("select title from gse where gse = ?", (gse_id,))
    gse_title = fix_text(c.fetchone()[0])

    keywords_list = [("and", and_keywords), ("or", or_keywords)]
    query = "select gse_gsm.gsm, gsm.title, gsm.source_name_ch1, gsm.description from gse_gsm, gsm " \
            "where gse_gsm.gse = ? and gse_gsm.gsm = gsm.gsm"
    if organism:
        query += " and lower(gsm.organism_ch1) = \"{}\"".format(organism.lower())

    for keyword_type, keywords in keywords_list:
        if keywords:
            query += " and "
            for i, keyword in enumerate(keywords.split(",")):
                if i == 0:
                    query += "("
                else:
                    query += " {} ".format(keyword_type)

                keyword = keyword.lower()
                query += "(lower(gsm.title) like \"%{keyword}%\" or lower(gsm.source_name_ch1) like \"%{keyword}%\" " \
                         "or lower(gsm.description) like \"%{keyword}%\" " \
                         "or lower(gsm.characteristics_ch1) like \"%{keyword}%\" " \
                         "or lower(gsm.treatment_protocol_ch1) like \"%{keyword}%\" " \
                         "or lower(gsm.extract_protocol_ch1) like \"%{keyword}%\")".format(keyword=keyword)

            query += ")"

    c.execute(query, (gse_id,))
    rows = c.fetchall()

    if not rows:
        query = "select gse_gsm.gsm, gsm.title, gsm.source_name_ch1, gsm.description from gse_gsm, gsm, gse " \
                "where gse_gsm.gse = ? and gse_gsm.gsm = gsm.gsm and gse_gsm.gse = gse.gse"
        if organism:
            query += " and lower(gsm.organism_ch1) = \"{}\"".format(organism.lower())

        for keyword_type, keywords in keywords_list:
            if keywords:
                query += " and "
                for i, keyword in enumerate(keywords.split(",")):
                    if i == 0:
                        query += "("
                    else:
                        query += " {} ".format(keyword_type)

                    keyword = keyword.lower()
                    query += "(lower(gse.title) like \"%{keyword}%\" or " \
                             "lower(gse.summary) like \"%{keyword}%\")".format(keyword=keyword)

                query += ")"

        c.execute(query, (gse_id,))
        rows = c.fetchall()

    conn.close()
    gsm_ids = []
    texts = []
    gsm_types = []

    for row in rows:
        gsm_title = row[1]
        gsm_type = clf.predict([gsm_title])[0]
        gsm_title = fix_text(gsm_title)

        text = gsm_title
        for t in row[2:]:
            if t:
                text += " " + t

        gsm_ids.append(row[0])
        texts.append(text)
        gsm_types.append(gsm_type)

    return gse_id, (gse_title, gsm_ids, texts, gsm_types)


def fix_text(text):
    """
    Replace alpha and beta symbol with words and add a space before alpha and beta
    Args:
        text: the text to be processed

    Returns:
        the processed text
    """
    for symbol, word in [("α", "alpha"), ("β", "beta"), ("γ", "gamma")]:
        text = re.sub(symbol, word, text)

    for word in ["alpha", "beta", "gamma"]:
        text = re.sub("\s*({})".format(word), r" \1", text, flags=re.IGNORECASE)

    return text


def filter_ctrl_pert(gse_gsm_info):
    """
    Filter the GSE that do not contain both control and perturbation samples
    Args:
        gse_gsm_info: the GSE and GSM info tuple

    Returns:
        True if there are both control and perturbation samples, False otherwise
    """
    gse_id, gsm_info = gse_gsm_info
    sample_types = gsm_info[3]
    has_ctrl = has_pert = False

    for sample_type in sample_types:
        if has_ctrl and has_pert:
            break

        if sample_type == "ctrl":
            has_ctrl = True
        elif sample_type == "pert":
            has_pert = True

    return has_ctrl and has_pert


def cluster_gsm(noble_coder, gse_gsm_info):
    gse_id, gsm_info = gse_gsm_info
    title, gsm_ids, texts, sample_types = gsm_info
    vectorizer = TfidfVectorizer(stop_words="english")
    x = vectorizer.fit_transform(texts)
    eps = MIN_EPS
    num_clusters = 0

    while num_clusters < 2 and eps <= MAX_EPS:
        db = DBSCAN(eps=eps, min_samples=2).fit(x)
        labels = db.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if num_clusters > 1:
            ctrl_clusters = defaultdict(list)
            pert_clusters = defaultdict(list)

            for i, label in enumerate(labels):
                if sample_types[i] == "ctrl":
                    ctrl_clusters[label].append(i)
                else:
                    pert_clusters[label].append(i)

            ctrl_indices = ctrl_clusters[min(ctrl_clusters.keys())]
            valid_pert_indices = [x for x in pert_clusters.values() if len(x) > 1]

            if len(ctrl_indices) > 1 or valid_pert_indices:
                results = []
                pert_agents = []

                for pert_indices in valid_pert_indices:
                    pert_agents.append(get_pert_agent(noble_coder, texts[pert_indices[0]], title))

                for ctrl_indices in ctrl_clusters.values():
                    for i, pert_indices in enumerate(valid_pert_indices):
                        result = (gse_id, pert_agents[i], tuple(gsm_ids), texts[ctrl_indices[0]],
                                  tuple(ctrl_indices)), (texts[pert_indices[0]], pert_indices)
                        results.append(result)

                return results
            else:
                num_clusters = 0

        eps += 0.1

    return []


def get_pert_agent(noble_coder, pert_text, title):
    pert_agent = run_noble_coder(pert_text, noble_coder)
    if pert_agent is None:
        pert_agent = run_noble_coder(title, noble_coder)

    if pert_agent is not None:
        for special_char in SPECIAL_CHARS:
            pert_agent = pert_agent.replace(special_char, " ")

        pert_agent = pert_agent.replace("Superfamily", "")
        mg = MyGeneInfo()
        response = mg.query(pert_agent)

        if response["hits"]:
            pert_agent = response["hits"][0]["symbol"]

    return pert_agent


def run_noble_coder(text, noble_coder):
    pert_agent = None
    with tempfile.TemporaryDirectory() as dirname:
        with open("{}/tmp.txt".format(dirname), "w") as f:
            f.write(text)

        command = "java -jar {noble_coder} -terminology NCI_Thesaurus " \
                  "-input {dirname} -output {dirname}".format(noble_coder=noble_coder, dirname=dirname)
        proc = Popen(shlex.split(command), stdout=PIPE, stderr=PIPE)
        out, err = [x.decode("utf-8") for x in proc.communicate()]

        if proc.returncode != 0 or "[Errno" in err:
            raise RuntimeError("Noble Coder failed to complete\nstdout: {}\n stderr {}".format(out, err))

        data = tablib.Dataset().load(open("{}/RESULTS.tsv".format(dirname)).read())
        for row in data:
            if "Amino Acid" in row[4] and "Pharmacologic Substance" not in row[4]:
                pert_agent = row[3]
                break

    return pert_agent


def get_target_pert_indices(gse_gsm_info):
    key, val = gse_gsm_info
    gse_id, pert_agent, gsm_ids, ctrl_text, ctrl_indices = key
    pert_texts, pert_indices = val
    target_index = max_score = max_days_diff = None

    ctrl_days_text = re.search("\d+\s*(d(ays?)?|h((ours?)|(r|rs)?))", ctrl_text, flags=re.IGNORECASE)
    pert_days_texts = [re.search("\d+\s*(d(ays?)?|h((ours?)|(r|rs)?))", x, flags=re.IGNORECASE) for x in pert_texts]

    if ctrl_days_text is None:
        ctrl_days_text = re.search("\d+", "0")

    if ctrl_days_text is not None and any(x is not None for x in pert_days_texts):
        ctrl_days_num = int(re.search("\d+", ctrl_days_text.group()).group())
        for i, pert_days_text in enumerate(pert_days_texts):
            if pert_days_text is not None:
                pert_days_num = int(re.search("\d+", pert_days_text.group()).group())
                days_diff = pert_days_num - ctrl_days_num

                if days_diff >= 0 and (max_days_diff is None or days_diff > max_days_diff):
                    max_days_diff = days_diff
                    target_index = i
    else:
        for i, pert_text in enumerate(pert_texts):
            score = jellyfish.jaro_winkler(ctrl_text, pert_text)
            if max_score is None or score > max_score:
                max_score = score
                target_index = i

    if target_index is None:
        return None

    gsm_ids = np.array(gsm_ids)
    ctrl_indices = list(ctrl_indices)
    target_pert_indices = pert_indices[target_index]

    microarray_grouping = np.chararray(len(gsm_ids), unicode=True)
    microarray_grouping[:] = "X"
    microarray_grouping[ctrl_indices] = "0"
    microarray_grouping[target_pert_indices] = "1"
    microarray_grouping = "".join(microarray_grouping)

    return gse_id, (pert_agent, "|".join(gsm_ids[ctrl_indices]), "|".join(gsm_ids[target_pert_indices]),
                    microarray_grouping)


def process_results(key_val):
    gse_id, results = key_val
    all_results = [gse_id]

    for result in results:
        all_results += list(result)

    headers = ["gse_id", "pert_agent", "ctrl_gsm_ids", "pert_gsm_ids", "microarray_grouping", "is_pert_title",
               "is_pert_summary", "organism", "direction"]
    df = pd.DataFrame.from_records([{headers[i]: all_results[i] for i in range(len(headers))}])
    df = df.set_index("gse_id")

    return df


def combine_results(df1, df2):
    return df1.append(df2)


# -----------------------------------------------------------------------------
# Merge functions
# -----------------------------------------------------------------------------
def merge_two_tuples(accum_tuple, new_tuple):
    """
    Merge two tuples
    Args:
        accum_tuple: the accumulated list of tuples
        new_tuple: the new tuple

    Returns:
        two new lists
    """
    l1, l2 = accum_tuple
    t1, t2 = new_tuple
    l1.append(t1)
    l2.append(t2)

    return l1, l2


def merge_two_lists(t1, t2):
    """
    Merge two tuples of lists
    Args:
        t1: the tuple of the first lists
        t2: the tuple of the second lists

    Returns:
        a new tuple of the merged lists
    """
    a1, b1 = t1
    a2, b2 = t2

    return a1 + a2, b1 + b2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("-d", "--db_file", required=True, help="The GEOmetadb file")
    required_args.add_argument("-t", "--gse_clf_title", dest="gse_title_pickle", required=True,
                               help="The GSE title classifier")
    required_args.add_argument("-s", "--gse_clf_summary", dest="gse_summary_pickle", required=True,
                               help="The GSE summary classifier")
    required_args.add_argument("-m", "--gsm_clf", dest="gsm_pickle", required=True, help="The GSM classifier")
    required_args.add_argument("-n", "--noble_coder", required=True, help="NobelCoder executable")

    parser.add_argument("-i", "--input_file", help="The list of GSE IDs")
    parser.add_argument("-ak", "--and_keywords", help="List of and keywords")
    parser.add_argument("-ok", "--or_keywords", help="List of or keywords")
    parser.add_argument("-o", "--organism", help="The organism")
    parser.add_argument("-p", "--min_partitions", type=int, default=2, help="The minimum number of partitions")
    parser.add_argument("-u", "--upload_loc", help="The s3 location for uploading the results")
    parser.add_argument("-r", "--region", default="us-west-2", help="AWS region (Default: %(default)s)")
    parser.set_defaults(method=main)

    args = parser.parse_args()
    if all(x is None for x in [args.input_file, args.and_keywords, args.or_keywords, args.organism]):
        parser.error("At least one of the following arguments are required: input file, keywords and organism")

    args.method(**vars(args))
