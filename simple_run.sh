#!/bin/bash

# simple_run.sh
# -------------
# A simple bash script to execute the script without having to type it out fully on command line.

model=gemma3:27b
dataset=hotpotqa
numtests=308


echo "python run.py --threshold --model $model --dataset $dataset --step True --num_test $numtests --k 1 --width 1 --depth 2 --debug 1"
# python run.py --model $model --dataset $dataset --step True --num_test $numtests --k 1 --width 1 --depth 2 --debug 1
python run.py --model $model --dataset $dataset --step True --num_test $numtests --k 2 --width 2 --depth 3 --debug 1

# the following is just to make it quicker to run these tests... I want to run some tests to see the accuracy of KG-RAG across the best parameters I found
# model=gpt-oss:120b
# python run.py --model $model --dataset $dataset --step True --num_test $numtests --kgrag_test 1 --k 2 --width 3 --depth 1
