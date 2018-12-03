#!/bin/sh
TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe train -solver solver_Lsfa_loss.prototxt  
echo 'Done.'
