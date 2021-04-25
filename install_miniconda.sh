#!/usr/bin/env bash

!(stat -t /usr/local/lib/*/dist-packages/google/colab > /dev/null 2>&1) && exit 

# change into working directory on colab
cd /content/

## install miniconda
curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
  "Miniconda3.sh"

chmod +x ./Miniconda3.sh
./Miniconda3.sh -b -f -p /usr/local

echo "Miniconda installed"
