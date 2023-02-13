#!/bin/bash

set -e

CONFIG="./config/sound2meg-phoneme.yaml"
DATA_ROOT="./data/MASC-MEG/"
OUTPUT="./data/formatted/words"
DATA_ID="250Hz-800ms"
LEVEL="word"
SUBJECT="12"

source venv/bin/activate

python src/preprocess.py $DATA_ROOT $OUTPUT -i $DATA_ID -s $SUBJECT -l $LEVEL -p $CONFIG --split --meg

python src/preprocess.py $DATA_ROOT $OUTPUT -i $DATA_ID -s $SUBJECT -l $LEVEL -p $CONFIG --split --audios

deactivate
