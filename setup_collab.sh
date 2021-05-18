#!/usr/bin/env bash

!(stat -t /usr/local/lib/*/dist-packages/google/colab > /dev/null 2>&1) && exit 

# change into working directory on colab
cd /content/

# setup python environment
conda create -n semantic_segmentation python=3.5 -y
source activate semantic_segmentation
conda install pip -y
pip install --upgrade pip
conda install pytorch=0.4.1 cuda90 -c pytorch -y
conda install -c pytorch torchvision -y
pip install -r requirements.txt
echo "python environment setup"

# download ADE20K dataset
chmod +x ./download_ADE20K.sh
./download_ADE20K.sh
echo "ADE20K dataset downloaded"

# remask dataset for training and validation
python ./remask_dataset/parse_training_set.py ./data/ADEChallengeData2016/annotations/training ./data/ADEChallengeData2016_remasked/annotations/training ./data/ADEChallengeData2016_remasked/training_remask.odgt -r
python ./remask_dataset/parse_training_set.py ./data/ADEChallengeData2016/annotations/validation ./data/ADEChallengeData2016_remasked/annotations/validation ./data/ADEChallengeData2016_remasked/validation_remask.odgt -r
echo "dataset remasked"

cp -r ./data/ADEChallengeData2016/images ./data/ADEChallengeData2016_remasked/
echo "images copied over to ./data/ADEChallengeData2016_remasked/"
