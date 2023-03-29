#!/bin/bash
if [[ $# -ne 2 ]]
then
    echo "Missing argument: data directory and/or output directory"
fi

source=$1
output=$2

for i in $source/*
do
    if echo $i | grep -q "sub"
    then
        echo "Found data source: $i"
        f=$(basename $i)
        subno=${f:3:2}
        echo sub-$subno >> $output
    fi
done
