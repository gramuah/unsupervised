#!/bin/bash

#Script to fetch all the pretrained models

#Models for UCF101
echo "Downloading pretrained models (3.1 GB)"

FILE=models.tar.gz
URL=http://agamenon.tsc.uah.es/Personales/rlopez/data/unsupervised/models.tar.gz


wget $URL -O $FILE

echo "Uncompressing..."

tar zxvf $FILE

echo "Done. Enjoy unsupervised learning!"
