#!/bin/bash

# Bash script for running the neural network training on CERN's LXPLUS system. Uses time and charge loss.
OUT_DIR=/eos/user/c/cristova/WaterCherenkov/Trials_Framework_20epoch_minFeatures/MultiGaus_qt_${1}
PREV_DIR=/eos/user/c/cristova/WaterCherenkov/Trials_Framework_10epoch_minFeatures/MultiGaus_qt_${1}


local_dir=${PWD}
echo "Copying input to " ${local_dir} 
xrdcp -r /eos/user/c/cristova/www/share/SK_Dataset/v1/*.h5 ${local_dir}/
echo "Making local output directory"
mkdir -p ${local_dir}/CRinGe_MultiGaus_qt_${1}

echo "Eval"
eval "$(conda shell.bash hook)"
echo "Activate"
conda activate /afs/cern.ch/work/c/cristova/.conda/envs/wcml_torch
echo "cd"
cd /afs/cern.ch/work/c/cristova/CRinGe_framework
echo "Starting"
python train_model.py -e 20 -b 200 -j 4 -t 0.75 -s 50000 -o ${local_dir}/CRinGe_MultiGaus_qt_${1} ${local_dir} _test.h5 --network_state ${PREV_DIR}/MultiGaus_qt_${1}/CRinGe_MultiGaus_qt_${1}/CRinGe_SK_MultiGaus.cnn --optimizer_state ${PREV_DIR}/CRinGe_MultiGaus_qt_${1}/CRinGe_SK_MultiGaus_optimizer.cnn CRinGe_SK_MultiGaus N_GAUS:${1} use_time:1 
#python train_model.py -e 10 -b 200 -j 4 -t 0.75 -s 50000 -o ${local_dir}/CRinGe_MultiGaus_qt_${1} ${local_dir} _test.h5 CRinGe_SK_MultiGaus N_GAUS:${1} use_time:1 
echo "DONE"

echo "COPYING OUTPUT"
mkdir -p ${OUT_DIR}
xrdcp -r ${local_dir}/CRinGe_MultiGaus_qt_${1} ${OUT_DIR}

echo "DELETING LOCAL OUTPUT"
rm -rf ${local_dir}/CRinGe_MultiGaus_qt_${1}

echo "Deleting input"
rm ${local_dir}/*.h5
