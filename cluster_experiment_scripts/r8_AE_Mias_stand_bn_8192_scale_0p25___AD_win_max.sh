#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..


mkdir -p /disk/scratch/${STUDENT_ID}/data/MiasPathological/
rsync -ua --progress --delete /home/${STUDENT_ID}/mlp_framework/data/MiasPathological/ /disk/scratch/${STUDENT_ID}/data/MiasPathological/
export DATASET_DIR=/disk/scratch/${STUDENT_ID}/data/

python anomaly_detection.py --experiment_name r8_AE_Mias_stand_bn_8192_scale_0p25___AD_win_max