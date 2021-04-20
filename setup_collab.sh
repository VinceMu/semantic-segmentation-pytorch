#!/usr/bin/env bash

!(stat -t /usr/local/lib/*/dist-packages/google/colab > /dev/null 2>&1) && exit 

# change into working directory on colab
cd /content/

## install miniconda
chmod +x ./install_miniconda.sh
./install_miniconda.sh

# setup python environment
conda create -n semantic_segmentation python=3.5 -y
conda install pip -y
conda install pytorch=0.4.1 cuda90 -c pytorch
conda install -c pytorch torchvision
pip install -r requirements.txt
source activate semantic_segmentation

# download ADE20K dataset
chmod +x ./download_ADE20K.sh
./download_ADE20K.sh

# remask dataset for training and validation
python ./remask_dataset/parse_training_set.py ./data/ADEChallengeData2016/annotations/training ./data/ADEChallengeData2016_remasked/annotations/training
python ./remask_dataset/parse_training_set.py ./data/ADEChallengeData2016/annotations/validation ./data/ADEChallengeData2016_remasked/annotations/validation
