import argparse
import os
import shlex
import shutil
from subprocess import Popen, PIPE
from pyspark import SparkContext, SparkConf
import pandas as pd
import subprocess
import boto3
import re

global parser_result

APPLICATION_FOLDER = "/mnt/app"
GENOME_REFERENCES_FOLDER = "/mnt/ref"
TEMP_OUTPUT_FOLDER = "/mnt/output"

star_collected_metrics = ["number of input reads", "uniquely mapped reads number", "number of splices: total",
                          "number of splices: annotated (sjdb)", "number of splices: gt/ag", "number of splices: gc/ag",
                          "number of splices: at/ac", "number of splices: non-canonical",
                          "number of reads mapped to multiple loci", "number of reads mapped to too many loci"]

picard_collected_metrics = ['pf_bases', 'pf_aligned_bases', 'ribosomal_bases', 'coding_bases', 'utr_bases',
                            'intronic_bases', 'intergenic_bases', 'ignored_reads', 'correct_strand_reads',
                            'incorrect_strand_reads']

hisat_ignore_metrics_pattern = r"^[\d.]+\%"
hisat_extract_metrics_pattern = r"(\d+)\s?(\([\d.]+\%\))?\s?(.*)\:?"


#################################
#  File splitting
#################################


def split_interleaved_file(file_prefix, file_content, output_dir):
    """
    Unpacks an interleaved file into the standard FASTQ format
    :param file_prefix: the prefix of the file name
    :param file_content: the lines of content from the input file
    :param output_dir: the location to store the unpacked files
    :return: a tuple with first element being a list of output file names
    (1 for se, 2 for pe); 2nd element a boolean flag - True if pe data,
    False otherwise
    """
    fastq_line_count_se = 4
    fastq_line_count_pe = 8
    paired_reads = False
    output_file_names = []

    file_prefix = output_dir + "/" + file_prefix
    output_file = file_prefix + "_1.fq"
    output_file_names.append(output_file)
    output_file_writer = open(output_file, 'w')

    count = 0
    for line in file_content.strip().split("\n"):
        # In the first line, check if it's paired or not
        if count == 0 and len(line.strip().split("\t")) == fastq_line_count_pe:
            paired_reads = True
            output_file_pair = file_prefix + "_2.fq"
            output_file_names.append(output_file_pair)
            output_pair_writer = open(output_file_pair, 'w')

        if paired_reads:
            parts = line.strip().split("\t")

            if len(parts) != fastq_line_count_pe:
                continue

            read_one = parts[:fastq_line_count_se]
            read_two = parts[fastq_line_count_se:]
            output_file_writer.write("\n".join(read_one) + "\n")
            output_pair_writer.write("\n".join(read_two) + "\n")
        else:
            output_file_writer.writelines(line.strip().replace("\t", "\n") + "\n")

        count += 1

    output_file_writer.close()
    if paired_reads:
        output_pair_writer.close()

    return output_file_names, paired_reads


def run_kallisto(file_names, output_dir):
    print("Aligning reads...")
    paired_read = True if len(file_names) == 2 else False
    sample_name = os.path.splitext(os.path.basename(file_names[0]))[0]
    kallisto_output_dir = "{}/{}".format(output_dir, sample_name)

    # try:
    if not os.path.exists(kallisto_output_dir):
        os.mkdir(kallisto_output_dir)
    # except FileExistsError:
    #     pass

    # Construct the shell command for executing the alignment tool
    aligner_args = "{app_folder}/kallisto quant {aligner_extra_args} -i {index_folder} -o {output_folder} " \
                   "{single_read_args} {fastq_file_names}". \
        format(app_folder=APPLICATION_FOLDER,
               aligner_extra_args="" if parser_result.aligner_extra_args is None else parser_result.aligner_extra_args,
               index_folder=GENOME_REFERENCES_FOLDER + "/kallisto_index/transcripts.idx",
               fastq_file_names=" ".join(file_names),
               output_folder=kallisto_output_dir,
               single_read_args="--single -l 200 -s 20" if not paired_read else "")
    print("Command: " + aligner_args)

    # Execute the shell command
    aligner_process = Popen(shlex.split(aligner_args), stdout=PIPE, stderr=PIPE)
    aligner_out, aligner_error = aligner_process.communicate()

    # Check for error using return code
    if aligner_process.returncode != 0:
        raise ValueError("Kallisto failed to complete (Non-zero return code)!\n"
                         "Kallisto stdout: {std_out} \Aligner stderr: {std_err}".format(std_out=aligner_out,
                                                                                        std_err=aligner_error))

    print('Completed reads alignment')

    counter_output = []
    with open(kallisto_output_dir + "/abundance.tsv") as f:
        for index, line in enumerate(f):
            if index == 0:  # Command summary and header
                continue

            line = line.strip().split()
            if len(line) == 0:
                print(line)
            gene, count = line[0], line[3]
            counter_output.append((sample_name + "\t" + gene, int(float(count))))

    return counter_output


#################################
#  Picard tools
#################################


def run_picard(sample_name, aligned_output_filepath, picard_output_dir):
    print("Getting alignment metrics...")
    picard_args = "java8 -jar {}/picard-tools/picard.jar CollectRnaSeqMetrics I={} O={}/output.RNA_Metrics " \
                  "REF_FLAT={}/refFlat.txt STRAND={} {}". \
        format(APPLICATION_FOLDER, aligned_output_filepath, picard_output_dir, GENOME_REFERENCES_FOLDER + "/genome_ref",
               parser_result.strand_specificity, parser_result.picard_extra_args)
    print("Command: " + picard_args)
    picard_process = Popen(shlex.split(picard_args), stdout=PIPE, stderr=PIPE)
    picard_out, picard_error = picard_process.communicate()

    if not os.path.isfile(picard_output_dir + "/output.RNA_Metrics"):
        raise ValueError("Picard tools failed to complete (No output file is found)!\n"
                         "Picard tools stdout: {} \nPicard tools stderr: {}".format(picard_out, picard_error))

    picard_qc_output = []
    with open(picard_output_dir + "/output.RNA_Metrics") as picard_qc:
        picard_lines = picard_qc.readlines()

        index = 0
        while index < len(picard_lines):
            current_line = picard_lines[index].strip()

            if current_line.startswith("##") and current_line[2:].strip().startswith("METRICS CLASS"):
                picard_metric_header = picard_lines[index + 1].strip().lower().split("\t")
                picard_metric_value = picard_lines[index + 2].strip().split("\t")

                metrics = dict(zip(picard_metric_header, picard_metric_value))
                for metric in picard_collected_metrics:
                    if metrics[metric] != "":
                        picard_qc_output.append((sample_name + "\t" + "QC_picard_" + metric, int(metrics[metric])))
                index += 2
            index += 1

    return picard_qc_output


def sum_gene_counts(cumulative_count, current_count):
    return cumulative_count + current_count


def set_gene_id_as_key(keyval):
    # Input: file_name\tgene, count as key,val
    # Output: file_name, (gene,count) as key,val
    key, val = keyval
    file_group, gene_id = key.split("\t")

    if gene_id == "QC_STAR_total_reads":
        print(keyval)

    return gene_id, [(file_group, val)]


def merge_count_by_gene_id(file_count_one, file_count_two):
    return file_count_one + file_count_two


def process_count_by_gene_id(keyval):
    gene_id, counts = keyval

    return pd.DataFrame({k: v for k, v in counts}, index=[gene_id])


def combine_gene_counts(df_one, df_two):
    return df_one.append(df_two)


#################################
#  Main functions
#################################


def alignment_count_step(keyval):
    # Input: file_name, file_content as key,val
    # Output: [sample_name\tgene, count] as [key,val]
    global parser_result, star_collected_metrics, picard_collected_metrics

    file_name, file_content = keyval
    prefix = file_name.rstrip("/").split("/")[-1].split(".")[0]
    sample_name = prefix.rsplit("_part", 1)[0]

    alignment_dir = TEMP_OUTPUT_FOLDER + "/alignment_" + prefix

    try:
        os.mkdir(alignment_dir)
    except:
        print('Alignment directory {} exist.'.format(alignment_dir))

    print("Recreating FASTQ file(s)")
    split_file_names, paired_reads = split_interleaved_file(prefix, file_content, alignment_dir)
    print("Recreating FASTQ file(s) complete. Files recreated: {}".format(",".join(split_file_names)))

    alignment_output_dir = alignment_dir + "/aligner_output"

    try:
        os.mkdir(alignment_output_dir)
    except:
        print('Alignment output directory {} exist.'.format(alignment_output_dir))

    # if parser_result.run_picard:
    #     picard_qc_output = run_picard(sample_name, aligned_output_filepath, alignment_output_dir)
    #     counter_output.extend(picard_qc_output)

    # shutil.rmtree(alignment_dir, ignore_errors=True)
    return run_kallisto(split_file_names, alignment_output_dir)


if __name__ == "__main__":
    global parser_result

    parser = argparse.ArgumentParser(description='Spark-based RNA-seq Pipeline')
    parser.add_argument('--input', '-i', action="store", dest="input_dir", help="Input directory - HDFS or S3")
    parser.add_argument('--output', '-o', action="store", dest="output_dir", help="Output directory - HDFS or S3")
    parser.add_argument('--annotation', '-a', action="store", dest="annotation_file",
                        help="Name of annotation file to be used")
    parser.add_argument('--strand_specificity', '-ss', action="store", dest="strand_specificity", nargs='?',
                        help="Strand specificity: NONE|FIRST_READ_TRANSCRIPTION_STRAND|SECOND_READ_TRANSCRIPTION_STRAND"
                        , default="NONE")
    parser.add_argument('--run_picard', '-rp', action="store_true", dest="run_picard", help="Run picard")
    parser.add_argument('--aligner_tools', '-at', action="store", dest="aligner", nargs='?',
                        help="Aligner to be used (STAR|HISAT2)", default="STAR")
    parser.add_argument('--aligner_extra_args', '-s', action="store", dest="aligner_extra_args", nargs='?',
                        help="Extra argument to be passed to alignment tool", default="")
    parser.add_argument('--counter_tools', '-ct', action="store", dest="counter", nargs='?',
                        help="Counter to be used (featureCount|StringTie)", default="featureCount")
    parser.add_argument('--counter_extra_args', '-c', action="store", dest="counter_extra_args", nargs='?',
                        help="Extra argument to be passed to quantification tool", default="")
    parser.add_argument('--picard_extra_args', '-p', action="store", dest="picard_extra_args", nargs='?',
                        help="Extra argument to be passed to picard tools", default="")
    parser.add_argument('--region', '-r', action="store", dest="aws_region", help="AWS region")

    parser_result = parser.parse_args()

    split_num = 0

    conf = SparkConf().setAppName("Spark-based RNA-seq Pipeline Multifile")
    sc = SparkContext(conf=conf)

    if parser_result.input_dir.startswith("s3://"):  # From S3

        s3_client = boto3.client('s3', region_name=parser_result.aws_region)
        # Get number of input files
        s3_paginator = s3_client.get_paginator('list_objects')
        input_bucket, key_prefix = parser_result.input_dir[5:].strip().split("/", 1)

        input_file_num = 0

        for result in s3_paginator.paginate(Bucket=input_bucket, Prefix=key_prefix):
            for file in result.get("Contents"):
                input_file_num += 1

        if input_file_num == 0:
            raise ValueError("Input directory is invalid or empty!")

        split_num = input_file_num
    else:  # From HDFS
        hdfs_process = Popen(shlex.split("hdfs dfs -count {}".format(parser_result.input_dir)),
                             stdout=PIPE, stderr=PIPE)
        hdfs_out, hdfs_error = hdfs_process.communicate()

        if hdfs_error:
            raise ValueError("Input directory is invalid or empty!")

        dir_count, file_count, size, path = hdfs_out.strip().split()

        split_num = int(file_count)

    input_files = sc.wholeTextFiles(parser_result.input_dir, split_num)

    count_output = input_files.flatMap(alignment_count_step).reduceByKey(sum_gene_counts)
    count_by_gene = count_output.map(set_gene_id_as_key).reduceByKey(merge_count_by_gene_id) \
        .map(process_count_by_gene_id)
    count_summary = count_by_gene.reduce(combine_gene_counts)

    count_qc_index = [f.startswith("QC_") for f in count_summary.index]
    count_only_index = [not x for x in count_qc_index]

    count_only_summary = count_summary[count_only_index]
    count_qc_summary = count_summary[count_qc_index]

    # If normalisation is required
    # count_summary = count_summary.apply(lambda x: x / np.sum(x) * 1000000)
    expressions_file = 'samples_expression.csv'
    qc_report_file = 'samples_qc_report.csv'
    count_only_summary = count_only_summary.sort_index()
    count_only_summary.to_csv(expressions_file)

    count_qc_summary = count_qc_summary.sort_index()
    count_qc_summary.to_csv(qc_report_file)

    if parser_result.input_dir.startswith("s3://"):  # From S3
        output_bucket, key_prefix = parser_result.output_dir.strip().strip("/")[5:].split("/", 1)
        s3_client.upload_file(expressions_file, output_bucket, key_prefix + "/" + expressions_file)
        s3_client.upload_file(qc_report_file, output_bucket, key_prefix + "/" + qc_report_file)
    else:
        subprocess.call(["hdfs", "dfs", "-mkdir", "-p", parser_result.output_dir.rstrip("/")])
        subprocess.call(["hdfs", "dfs", "-put", expressions_file, parser_result.output_dir.rstrip("/") + "/"
                         + expressions_file])
        subprocess.call(["hdfs", "dfs", "-put", qc_report_file, parser_result.output_dir.rstrip("/") + "/"
                         + qc_report_file])

    os.remove(expressions_file)
    os.remove(qc_report_file)
