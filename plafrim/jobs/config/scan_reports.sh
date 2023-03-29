#!/bin/bash
if [[ $# -ne 3 ]]
then
    echo "Missing argument: data directory and/or output directories"
    exit 1
fi

source=$1
output=$2
diffoutput=$3

for i in $source/sub-*-predictions/*
do  
    if [[ -f $i/r2s.npy ]]
    then
        echo "Found report: $i"
    
        if echo $i | grep -q "concat"
        then
            echo $i >> $diffoutput
        else
            echo $i >> $output
        fi
    else
        continue
    fi
done
