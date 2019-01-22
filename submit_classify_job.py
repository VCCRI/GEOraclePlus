import configparser
import argparse
import boto3
import utility
import sys
from collections import OrderedDict


SPARK_EXTRA_CONFIG = [("spark.python.profile", "true"),
                      ("spark.python.worker.reuse", "false"),
                      ("spark.yarn.executor.memoryOverhead", "4096"),
                      ("spark.driver.maxResultSize", "3g"),
                      ("spark.executor.extraJavaOptions",
                       "-Dlog4j.configuration=file:///etc/spark/conf/log4j.properties "
                       "-XX:+UseConcMarkSweepGC -XX:CMSInitiatingOccupancyFraction=30 "
                       "-XX:MaxHeapFreeRatio=50 -XX:+CMSClassUnloadingEnabled "
                       "-XX:MaxPermSize=512M -XX:OnOutOfMemoryError='kill -9 %%p'"
                       " -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/mnt/app/oom_dump_`date`.hprof")]


def submit_classify_job(job_config, cluster_id, dry_run, **kwargs):
    job_configuration = "config/classify_job.config"
    if job_config is not None and job_config.strip() != "":
        job_configuration = job_config.strip()

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(job_configuration)

    if cluster_id is None or cluster_id.strip() == "":
        cluster_id = utility.get_cluster_id(dry_run)
    else:
        cluster_id = cluster_id.strip()

    if cluster_id != "" and check_configuration(config):
        if config["job_config"].get("upload_classify_script", "False") == "True":
            utility.upload_files_to_s3([(config["job_config"]["classify_script"],
                                         config["job_config"]["classify_script_local_location"],
                                         config["job_config"]["classify_script_s3_location"])], dry_run)

        num_executors = calculate_num_executor(cluster_id, config["spark_config"]["executor_memory"])
        if num_executors < 0:
            config["spark_config"]["num_executors"] = "None"
        else:
            config["spark_config"]["num_executors"] = str(num_executors)

        config["spark_config"]["executor_cores"] = "1"

        job_argument = build_command(cluster_id, config, num_executors)

        if not dry_run:
            emr_client = boto3.client("emr")
            # warn user before removing any output
            out = config["script_arguments"]["output_location"]
            # find out which output dirs, if any, exist
            dirs_to_remove = utility.check_s3_path_exists([out])
            # create a list of the names of the directories to remove
            if dirs_to_remove:
                response = input("About to remove any existing output directories." +
                                 "\n\n\t{}\n\nProceed? [y/n]: ".format(
                                     '\n\n\t'.join(dirs_to_remove)))
                while response not in ['y', 'n']:
                    response = input('Proceed? [y/n]: ')
                if response == 'n':
                    print("Program Terminated.  Modify config file to change " +
                          "output directories.")
                    sys.exit(0)
                # remove the output directories
                if not utility.remove_s3_files(dirs_to_remove):
                    print("Program terminated")
                    sys.exit(1)
            job_submission = emr_client.add_job_flow_steps(**job_argument)
            print("Submitted job to cluster {}. Job id is {}".format(cluster_id, job_submission["StepIds"][0]))
        else:
            print(job_argument)


def check_configuration(config):
    if not utility.check_config(config, "job_config", ["name", "action_on_failure", "classify_script",
                                                       "classify_script_s3_location", "upload_classify_script"]):
        return False

    if not utility.check_upload_config(config["job_config"], "upload_classify_script", "classify_script",
                                       "classify_script_local_location", "classify_script_s3_location"):
        return False

    if not utility.check_config(config, "spark_config", ["driver_memory", "executor_memory"]):
        return False

    if not utility.check_config(config, "script_arguments", ["output_location", "region"]):
        return False

    if not utility.check_s3_region(config["script_arguments"]["region"]):
        return False

    if all(x not in config["script_arguments"] for x in ["input_location", "organism", "and_keywords", "or_keywords"]):
        print("At least one of the following script arguments are required: input_location, organism, and_keywords, "
              "or_keywords")
        return False

    return True


def calculate_num_executor(cluster_id, executor_memory):
    global SPARK_EXTRA_CONFIG

    memory_overhead = 512
    for conf in SPARK_EXTRA_CONFIG:
        if conf[0] == "spark.yarn.executor.memoryOverhead":
            memory_overhead = int(conf[1])

    memory_per_executor = int(executor_memory.strip("m")) / 1024 + memory_overhead / 1024

    total_mem, total_cpu = utility.get_cluster_mem_cpu(cluster_id)

    if total_mem < 0 or total_cpu < 0:
        num_executors = -1  # dry run
    else:
        num_executors = int(total_mem / memory_per_executor)

    return num_executors


def build_command(cluster_id, config, num_executors):
    job_arguments = OrderedDict()
    job_arguments["JobFlowId"] = cluster_id

    step_arguments = OrderedDict()
    step_arguments['Name'] = config["job_config"]["name"]
    step_arguments["ActionOnFailure"] = config["job_config"]["action_on_failure"]

    hadoop_arguments = OrderedDict()
    hadoop_arguments["Jar"] = "command-runner.jar"

    command_args = ["spark-submit",
                    "--deploy-mode", "cluster"]

    for config_name, config_value in SPARK_EXTRA_CONFIG:
        command_args.append("--conf")
        command_args.append("{}={}".format(config_name, config_value))

    for spark_conf in config["spark_config"]:
        command_args.append("--" + spark_conf.replace("_", "-"))
        command_args.append(config["spark_config"][spark_conf])

    command_args.append(config["job_config"]["classify_script_s3_location"].rstrip("/") + "/" +
                        config["job_config"]["classify_script"])

    command_args.append("-u")
    command_args.append(config["script_arguments"]["output_location"])
    command_args.append("-d")
    command_args.append("/mnt/clf_data/GEOmetadb.sqlite")
    command_args.append("-t")
    command_args.append("/mnt/clf_data/gse_clf_title.pickle")
    command_args.append("-s")
    command_args.append("/mnt/clf_data/gse_clf_summary.pickle")
    command_args.append("-m")
    command_args.append("/mnt/clf_data/gsm_clf.pickle")
    command_args.append("-n")
    command_args.append("/mnt/app/NobleCoder.jar")
    command_args.append("-p")
    command_args.append(str(num_executors))
    command_args.append("-r")
    command_args.append(config["script_arguments"]["region"])

    if "input_location" in config["script_arguments"]:
        command_args.append("-i")
        command_args.append(config["script_arguments"]["input_location"])

    if "organism" in config["script_arguments"]:
        command_args.append("-o")
        command_args.append(config["script_arguments"]["organism"])

    if "and_keywords" in config["script_arguments"]:
        command_args.append("-ak")
        command_args.append(config["script_arguments"]["and_keywords"])

    if "or_keywords" in config["script_arguments"]:
        command_args.append("-ok")
        command_args.append(config["script_arguments"]["or_keywords"])

    hadoop_arguments['Args'] = command_args
    step_arguments["HadoopJarStep"] = hadoop_arguments
    job_arguments["Steps"] = [step_arguments]

    return job_arguments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Job submission script for Classifying GSE')
    parser.add_argument('--config', '-c', action="store", dest="job_config", default="", help="Job configuration file")
    parser.add_argument('--cluster-id', '-id', action="store", dest="cluster_id", help="Cluster ID for submission")
    parser.add_argument('--dry-run', '-d', action="store_true", dest="dry_run",
                        help="Produce the configurations for the job flow to be submitted")
    parser.set_defaults(method=submit_classify_job)

    parser_result = parser.parse_args()
    parser_result.method(**vars(parser_result))
