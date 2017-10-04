#!/usr/bin/env sh

python preprocess.py
python plot_histogram_of_sentence_length.py

GPU=0
CONFIG=./config/template.ini

python main.py \
    --gpu $GPU \
    --config $CONFIG \
    --mode train

python main.py \
    --gpu $GPU \
    --config $CONFIG \
    --mode analysis

