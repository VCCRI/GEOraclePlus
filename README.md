# GEOracle+

An automated, fast and accurate tool that utilises cloud computing and machine learning to perform data mining in the GEO database

## Prerequisites

Python3 is required, install the required Python packages.

`pip install -r requirements.txt`

[PySpark](https://spark.apache.org/) is also required, follow the instructions [here](https://spark.apache.org/downloads.html) to install.

Install the required R packages.

`Rscript install_packages.R`

Download the required databases and tools.

`bash dl_db_tools.sh <output_dir>`

## Usage

The pipeline contains two steps - classification and analysis. The classification step identifies perturbation experiments, groups replicate samples and matches control and perturbation samples. The analysis step calculates differential expression using the classification results from the previous step.

### Running the scripts locally

#### Classification step

The classification step requires either a file containing a list of GSE IDs by using the `-i` option, or some keywords to search through GEO using the `-ak`, `-ok` and/or `-o` options.

Below is an example of the format of the file containing a list of file.

```
GSE14491
GSE16416
GSE17708
GSE23952
GSE28448
GSE42373
```

Example command for running the classification step.

```
spark-submit source/classify_gse/classify_gse.py \
    -i gse_ids.txt \
    -d GEOmetadb.sqlite \
    -t classifiers/gse_clf_title.pickle \
    -s classifiers/gse_clf_summary.pickle \
    -m classifiers/gsm_clf.pickle \
    -n NobleCoder-1.0.jar
```

Detailed script usage can be access using the `-h` option.

#### Analysis step

The analysis step requires the output file from the previous step.

Example command for running the analysis step.

```
spark-submit source/analyse_gse/analyse_gse.py \
    -i microarray_classified_results.csv \
    -d GEOmetadb.sqlite \
    -ms source/analyse_gse/analyse_microarray.R \
    -rs source/analyse_gse/analyse_rna_seq.R \
    -mm mouse_matrix.h5 \
    -hm human_matrix.h5
```

Detailed script usage can be access using the `-h` option.

### Running on AWS

#### AWS Command Line Interface
The AWS Command Line Interface (AWS CLI) client is a utility from which the user can interface with the AWS "universe":
list objects in an S3 database, check on a running EMR cluster, and many other functions.

Instructions on how to install the AWS CLI can be found at:
http://docs.aws.amazon.com/cli/latest/userguide/installing.html

In a Linux environment, the command to install the AWS CLI is:

`pip install awscli`

Once installed, the AWS CLI needs to be configured with your AWS credentials.

#### AWS Credentials
Running _GEOracle+_ on AWS requires the user to have an AWS account, and to additionally have created an _AWS Access Key_.  The _AWS
Access Key_ enables programmatic access to AWS resources.  For information on how to create an AWS Access key,
refer to:

[Managing Access Keys for IAM Users](http://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html)

When you have your AWS _Access Key ID_ and _Secret Access Key_, you should configure your AWS CLI, by typing the
following command:

`aws configure`

Enter the information as prompted, including the _Default region name_.  You may leave the _Default output format_ by
pressing `Enter` if you are unsure about this field.

Once this configuration is complete, you can use the AWS CLI by typing something like:

`aws help`

#### Submitting jobs on AWS

Configure `config/emr_cluster.config` file for launching cluster. And also configure `config/classify_job.config` and `config/de_analysis_job.config` files for submitting classification and analysis jobs respectively.

After that, run `python launch_cluster.py --config config/emr_cluster.config` to launch a cluster. Wait until the cluster has been created successfully before running `python submit_classify_job.py --config config/classify_job.config` and `python submit_de_analysis_job.py --config config/de_analysis_job.config` to submit classification and analysis jobs respectively.

The output files will be stored on the S3 bucket specified in the configuration files.