#!/bin/bash

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
python train_model.py -e 10 -b 200 -j 4 -t 0.75 -s 5000 -o ${local_dir}/CRinGe_MultiGaus_qt_${1} ${local_dir} _test.h5 CRinGe_SK_MultiGaus N_GAUS:${1} use_time:1
echo "DONE"

echo "COPYING OUTPUT"
mkdir -p /eos/user/c/cristova/WaterCherenkov/Trials_Framework/MultiGaus_qt_${1}
xrdcp -r ${local_dir}/CRinGe_MultiGaus_qt_${1} /eos/user/c/cristova/WaterCherenkov/Trials_Framework/MultiGaus_qt_${1}

echo "DELETING LOCAL OUTPUT"
rm -rf ${local_dir}/CRinGe_MultiGaus_qt_${1}

echo "Deleting input"
rm ${local_dir}/*.h5
