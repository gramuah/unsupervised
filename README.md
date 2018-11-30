# Unsupervised learning from videos using temporal coherency deep networks

## Introduction

This is a repository with an implementation of the unsupervised learning solutions described in our [CVIU  journal paper](https://arxiv.org/abs/1801.08100). 


We provide here the pre-trained models needed to reproduce all the experiments detailed in the paper.

### License

This repository is released under the GNU General Public License v3.0 License (refer to the LICENSE file for details).

### Citing

If you make use of this data and software, please cite the following reference in any publications:

	@Article{Redondo-Cabrera2018,
	author 	= {Redondo-Cabrera, C. and Lopez-Sastre, R.~J.},
	title   = {Unsupervised learning from videos using temporal coherency deep networks},
	journal = {CVIU},
	year    = {2018},
	}


## Requirements

The has been developed and tested under Ubuntu 14.04 and Ubuntu 16.04.

 A Caffe installation is required.


## Pre-trained Unsupervised models

We provide pre-trained Caffe models trained on the two datasets used in our discovery experiments: the UCF-101 and the 5-Context.


Note that we always release the models for the two loss functions for unsupervised learning introduced in the paper, i.e. the Lq and Ls losses.

Python script are provided with all the new layers implemented.

## Extract features demo

We also release a demo to show how to extract features from a set of video frames with any of our pretrained models.


```Shell
    cd extract_features_demo
    python 
```


