# Unsupervised learning from videos using temporal coherency deep networks

## Introduction

This is a repository with an implementation of the unsupervised learning solutions described in our [CVIU  journal paper](https://doi.org/10.1016/j.cviu.2018.08.003). 


We provide here the pre-trained models needed to reproduce all the experiments detailed in the paper.

### License

This repository is released under the GNU General Public License v3.0 License (refer to the LICENSE file for details).

### Citing

If you make use of this data and software, please cite the following reference in any publications:

	@Article{Redondo-Cabrera2018,
	author 	= {Redondo-Cabrera, C. and Lopez-Sastre, R.~J.},
	title   = {Unsupervised learning from videos using temporal coherency deep networks},
	volume = {179},
	pages = {79-89},
	year = {2019},
	issn = {1077-3142},
	doi = {https://doi.org/10.1016/j.cviu.2018.08.003},
	journal = {CVIU},
	}


## Requirements

The project has been developed and tested under Ubuntu 14.04 and Ubuntu 16.04.

A [Caffe](http://caffe.berkeleyvision.org/) installation is required.


## Pre-trained Unsupervised models

We provide pre-trained Caffe models trained on the two datasets used in our discovery experiments: UCF-101 and 5-Context.

Note that we always release the models for the two loss functions for unsupervised learning introduced in the paper, i.e. the Lq and Ls losses.

Python script are provided with all the new layers implemented.


## Extract features demo

We also release a demo to show how to extract features from a set of video frames with any of our pretrained models.

```Shell
    cd extract_features_demo
    python extract_features.py
```

In the script extract_features.py simply change the following lines to select one of our pre-trained models for the feature extraction:

```Shell
    PRETRAINED_FILE = 'path_to_one_of_our_caffe_models.caffemodel'
    MODEL_FILE = 'path_to_the_correspoding_deploy.prototxt'
```

