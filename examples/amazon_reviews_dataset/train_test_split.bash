#!/bin/bash
# Use to split a fasttext corpus into a train and test set.
# Takes two arguments:
#  arg 1 - integer N > 1, every Nth item goes to test set
#  arg 2 - string containing the name of the fasttest preprocessed corpus

awk "NR % $1 != 0" $2 > train_$1_$2
awk "NR % $1 == 0" $2 > test_$1_$2
