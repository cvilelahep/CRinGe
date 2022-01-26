#!/bin/bash
########################################################
# 
# Purpose: Job submission script for trainning GeNN
#
#
# Or interactive test job with: 
#          salloc -c 4 --mem 32000M --gres=gpu:1
#
########################################################

#SBATCH --job-name=CRinGe

#SBATCH --gres=gpu:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=9         # CPU cores/threads
#SBATCH --mem=32000M              # memory per node
#SBATCH --time=1-00:00            # time (DD-HH:MM)

#SBATCH --account=def-pdeperio

#SBATCH --output=log-CRinGe-%u-%J.txt
#SBATCH --error=log-CRinGe-%u-%J.txt

# exit when any command fails
set -e

PROJECT_DIR=`readlink -f /project/6008045/junjiex/CRinGe`

npeak=(1 2 3 4 5 6 7 8 9 10)
indir="${PROJECT_DIR}/framework_training_output" # this should be the dir of your .cnn files
datadir="${PROJECT_DIR}/SKML/v1" # this should be your dir of MC data
#define output dir
outdir="${PROJECT_DIR}/framework_scan_output"
outlogdir="${PROJECT_DIR}/framework_scan_log"
mkdir -p "$outdir"
mkdir -p "$outlogdir"

#where the exec locates
bin="${PROJECT_DIR}/CRinGe.Fork/CRinGe"
train_models=("CRinGe_SK_MultiGaus")
time_flags=(0 1)
models_name=( "MultiGaus" "MultiGausTime" )

echo "$train_model"

for tf in "${time_flags[@]}"; do
    for model in "${train_models[@]}"; do
	for np in "${npeak[@]}"; do
	    #echo "${models_name[$tf]}"
	    JOBTIME=`date` sbatch --gres=gpu:1 --cpus-per-task=6  --time=1-00:00 --account=def-pdeperio --mem=32000M --output="${outlogdir}/log-${models_name[$tf]}${np}-factorized.txt" --error="${outlogdir}/log-${models_name[$tf]}${np}-factorized.txt" --job-name="${models_name[$tf]}_${np}" "${PROJECT_DIR}/enter_container.sh" "python ${bin}/scan_loss.py -i ${indir}/CRinGe_SK_${models_name[$tf]}_${np} -f 1 -j 4 -n 10 -p 1 -o ${outdir}/CRinGe_SK_${models_name[$tf]}_${np} ${datadir} $model N_GAUS:${np} use_time:${tf}"
	done
    done
done
