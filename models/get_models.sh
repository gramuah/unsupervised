#!/bin/bash

#Script to extract all the pretrained models

#Models for UCF101
FILE=UCF101.tar.gz
echo "Uncompressing UCF101 dataset models..."
tar zxvf $FILE


FILE=5-Contexts.tar.gz
echo "Uncompressing 5-Contexts dataset models..."
tar zxvf $FILE

echo "Done. Enjoy unsupervised learning!"
