#!/usr/bin/env bash

usage() {
    printf "usage: $0 s3://mybucket/destination-key\n"
    exit 1
}

# exit on any failure
set -e

# check have one argument that starts with "s3://"
[[ $# -eq 1 && ${1:0:5} == "s3://" ]] || usage

s3_uri=$1

# create a temporary directory
tmp=tmp-$( date "+%s" )
mkdir $tmp
cd $tmp

# GEOmetadb
wget http://starbuck1.s3.amazonaws.com/sradb/GEOmetadb.sqlite.gz

# NCI Terminologies
wget http://noble-tools.dbmi.pitt.edu/data/NCI_Thesaurus.term.zip

# ARCHS4
wget https://s3.amazonaws.com/mssm-seq-matrix/human_matrix.h5
wget https://s3.amazonaws.com/mssm-seq-matrix/mouse_matrix.h5

cd ..
aws s3 sync $tmp $s3_uri
rm -r $tmp

printf "DB FILES SUCCESSFULLY DOWNLOADED AND COPIED TO S3\n"
