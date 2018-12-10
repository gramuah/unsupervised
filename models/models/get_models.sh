#!/bin/bash

#Script to fetch all the pretrained models

#Models for UCF101
echo "Downloading pretrained models (3.1 GB)"


FILE=UCF101.tar.gz
URL=http://agamenon.tsc.uah.es/Personales/rlopez/data/unsupervised/UCF101.tar.gz
wget $URL -O $FILE
echo "Uncompressing UCF101 dataset models..."
tar zxvf $FILE


FILE=5-Contexts.tar.gz
URL=http://agamenon.tsc.uah.es/Personales/rlopez/data/unsupervised/5-Contexts.tar.gz
wget $URL -O $FILE
echo "Uncompressing 5-Contexts dataset models..."
tar zxvf $FILE





echo "Done. Enjoy unsupervised learning!"
