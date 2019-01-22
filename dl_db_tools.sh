#!/usr/bin/env bash

usage() {
    printf "usage: $0 <output_dir>"
    exit 1
}

# exit on any failure
set -e

# check have one argument that starts with "s3://"
[[ $# -eq 1 ]] || usage

output_dir=$1
mkdir -p $output_dir
cd $output_dir

# GEOmetadb
wget http://starbuck1.s3.amazonaws.com/sradb/GEOmetadb.sqlite.gz
gunzip GEOmetadb.sqlite.gz

# NCI Terminologies
wget http://noble-tools.dbmi.pitt.edu/data/NCI_Thesaurus.term.zip

# unzip terminologies
unzip NCI_Thesaurus.term.zip
mkdir -p ~/.noble
mkdir -p ~/.noble/terminologies
mv NCI_Thesaurus.term ~/.noble/terminologies/

# ARCHS4
wget https://s3.amazonaws.com/mssm-seq-matrix/human_matrix.h5
wget https://s3.amazonaws.com/mssm-seq-matrix/mouse_matrix.h5

# Nobel Coder
wget https://github.com/dbmi-pitt/nobletools/releases/download/1.0/NobleCoder-1.0.jar

printf "Required databases and tools have been downloaded\n"
