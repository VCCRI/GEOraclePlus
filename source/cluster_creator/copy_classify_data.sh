#!/bin/bash
# copies reference files from S3 & unzips
# input args:
#   $1 - AWS S3 URI for location containing genome reference, star index and hisat index

# want to terminate on error
set -e
set -o pipefail

clf_data_dir=/mnt/clf_data

aws s3 sync $1 $clf_data_dir

pushd $clf_data_dir
# unzip any .gz files in current directory or any subdirectories
# determine if there are any .gz files; note that without this test, the xargs command would fail with a null input
zip_files=$( find -L . -name "*.gz" -print0 )
if [ "$zip_files" != "" ] ; then
    # unzip all the .gz files using as many processors as possible
    find -L . -name "*.gz" -print0 | xargs -0 -n1 -P0 gunzip
fi

# unzip terminologies
unzip NCI_Thesaurus.term.zip
mkdir /var/lib/hadoop-yarn/
mkdir /var/lib/hadoop-yarn/.noble
mkdir /var/lib/hadoop-yarn/.noble/terminologies
mv NCI_Thesaurus.term /var/lib/hadoop-yarn/.noble/terminologies/
popd

