#!/bin/bash

N_GAUS=5

OUT_DIR=/eos/user/c/cristova/WaterCherenkov/Reco_qt_N_GAUS_${N_GAUS}/${1}/

local_dir=${PWD}
echo "Copying input to " ${local_dir} 
xrdcp -r /eos/user/c/cristova/www/share/SK_Dataset/v1/*${1}-_npztoh5_test.h5 ${local_dir}/
echo "Making local output directory"
mkdir -p ${local_dir}/CRinGe_MultiGaus_qt_${1}

echo "Eval"
eval "$(conda shell.bash hook)"
echo "Activate"
conda activate /afs/cern.ch/work/c/cristova/.conda/envs/wcml_torch
echo "cd"
cd /afs/cern.ch/work/c/cristova/CRinGe_framework

FRAC=`bc -l <<< "0.75 + ${2} / 2000"`

echo "Starting"

python fit_event.py ${local_dir} ${1}-_npztoh5_test.h5 500 -t ${FRAC} reco_${1}_${2}.h5 CRinGe_SK_MultiGaus /eos/user/c/cristova/WaterCherenkov/Trials_Framework_10epoch_minFeatures/MultiGaus_qt_${N_GAUS}/CRinGe_MultiGaus_qt_${N_GAUS}/CRinGe_SK_MultiGaus.cnn N_GAUS:${N_GAUS} use_time:1

echo "DONE"

echo "COPYING OUTPUT"
mkdir -p ${OUT_DIR}
xrdcp -r ${local_dir}/reco_${1}_${2}.h5 ${OUT_DIR}

echo "DELETING LOCAL INPUTS and OUTPUT"
rm ${local_dir}/*.h5
