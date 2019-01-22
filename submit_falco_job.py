import argparse

from submit_download_job import submit_download_job
from submit_analysis_job import submit_analysis_job


def main(download_config, analysis_config, cluster_id, dry_run, **kwargs):
    submit_download_job(download_config, cluster_id, dry_run)
    submit_analysis_job(analysis_config, cluster_id, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Job submission script for spark-based RNA-seq Pipeline')
    parser.add_argument('--download_config', '-dc', action="store", default="", help="Download job configuration file")
    parser.add_argument('--analysis_config', '-ac', action="store", default="", help="Analysis job configuration file")
    parser.add_argument('--cluster-id', '-id', action="store", dest="cluster_id", help="Cluster ID for submission")
    parser.add_argument('--dry-run', '-d', action="store_true", dest="dry_run",
                        help="Produce the configurations for the job flow to be submitted")
    parser.set_defaults(method=main)

    parser_result = parser.parse_args()
    parser_result.method(**vars(parser_result))

