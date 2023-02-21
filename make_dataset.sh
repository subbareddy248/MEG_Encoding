#!/bin/bash

set -e

DATA_ROOT="./data/MASC-MEG/"
SUBJECT="12"

source venv/bin/activate

CONFIG="./config/decim-4-no-baseline.yaml"
DATA_ID="250Hz-800ms-no-baseline"
LEVEL="phoneme"
OUTPUT="./data/formatted/$LEVEL"

python src/preprocess.py $DATA_ROOT $OUTPUT -i $DATA_ID -s $SUBJECT -l $LEVEL -p $CONFIG --split --meg

CONFIG="./config/decim-4-no-baseline.yaml"
LEVEL="word"
DATA_ID="250Hz-800ms-no-baseline"
OUTPUT="./data/formatted/$LEVEL"

python src/preprocess.py $DATA_ROOT $OUTPUT -i $DATA_ID -s $SUBJECT -l $LEVEL -p $CONFIG --split --meg


CONFIG="./config/decim-10-no-baseline.yaml"
LEVEL="phoneme"
DATA_ID="100Hz-800ms-no-baseline"
OUTPUT="./data/formatted/$LEVEL"

python src/preprocess.py $DATA_ROOT $OUTPUT -i $DATA_ID -s $SUBJECT -l $LEVEL -p $CONFIG --split --meg


CONFIG="./config/decim-10-no-baseline.yaml"
LEVEL="word"
DATA_ID="100Hz-800ms-no-baseline"
OUTPUT="./data/formatted/$LEVEL"

python src/preprocess.py $DATA_ROOT $OUTPUT -i $DATA_ID -s $SUBJECT -l $LEVEL -p $CONFIG --split --meg

#python src/preprocess.py $DATA_ROOT $OUTPUT -i $DATA_ID -s $SUBJECT -l $LEVEL -p $CONFIG --split --audios

deactivate

