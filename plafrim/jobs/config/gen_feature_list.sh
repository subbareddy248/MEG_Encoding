#!/bin/bash
if [[ $# -ne 2 ]]
then
    echo "Missing argument: data directory and/or output directory"
    exit 1
fi

source=$1
output=$2

for i in $source/*
do
    echo "Found feature source: $i"
    if echo $i | grep -q ".npy"
    then
        basename $i >> $output
    fi
done
