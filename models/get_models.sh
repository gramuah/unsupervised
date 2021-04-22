#!/bin/bash

#Script to fetch all the pretrained models

#Models for UCF101
echo "Downloading pretrained models (3.1 GB)"


FILE=UCF101.tar.gz
URL=https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/EbLgFp1rKbVAhOipRtGXdAwBAwvfGCBfRxBcC7wez0rwIw?&Download=1
wget $URL -O $FILE
echo "Uncompressing UCF101 dataset models..."
tar zxvf $FILE


FILE=5-Contexts.tar.gz
URL=https://universidaddealcala-my.sharepoint.com/:u:/g/personal/gram_uah_es/ERCg69mZP4VDpXcUZN8lBJgBIHxDaHEX7zK558yk6hQhrQ?&Download=1
wget $URL -O $FILE
echo "Uncompressing 5-Contexts dataset models..."
tar zxvf $FILE





echo "Done. Enjoy unsupervised learning!"
