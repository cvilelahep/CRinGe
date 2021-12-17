#!/bin/bash

# Example script to train model on Stony Brook NN group's ivy machine
python train_model.py -e 5 -b 150 -j 4 -t 0.75 -s 1000 -o ${PWD}/test/ /storage/shared/mojia/trainingSamples/ _npztoh5_test.h5 CRinGe_SK_MultiGaus N_GAUS:6