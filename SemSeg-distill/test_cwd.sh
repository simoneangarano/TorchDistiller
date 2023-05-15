#!/bin/bash

CITYSCAPES=../../CIRKD/data/Cityscapes
NORM=none
DIVERGENCE=mse
KD_WEIGHT=1
CWD_WEIGHT=3
TEMPERATURE=4
KD=True
CWD=True
ADV=False
FEAT=False
VERSION=vS
GPU=0

NAME=${KD}_${ADV}_${FEAT}_${NORM}_${DIVERGENCE}_${WEIGHT}_${TEMPERATURE}_${VERSION}

python val.py --data_dir $CITYSCAPES \
               --restore_from ckpt/$NAME/city_39999_G.pth \
               --gpu $GPU
              
#python test.py --data_dir $CITYSCAPES \
#               --restore_from ckpt/$NAME/city_39999_G.pth \
#               --gpu $GPU