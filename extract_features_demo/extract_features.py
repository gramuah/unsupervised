#!/usr/bin/env python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import io
from PIL import Image
import scipy.misc
import scipy.io as sio
import time
import pdb
import glob
import pickle as pkl
import random
import h5py
import skimage.io
import copy

# Make sure that caffe is on the python path:
caffe_root = '{caffe_root}/python'  # this file is expected to be in {caffe_root}/python
import sys
sys.path.insert(0, caffe_root)

import caffe

def main():
    
    print('Drawing net to %s' % MODEL_FILE)
    # decrease if you want to preview during training
    PRETRAINED_FILE = 'path_to_one_of_our_caffe_models.caffemodel' # IMPORTANT: You need to select a pretrained model	
    MODEL_FILE = 'path_to_the_correspoding_deploy.prototxt' # IMPORTANT: You need to select a the corresponding deploy.prototxt	
    caffe.set_mode_cpu()
    caffe.set_device(1)
    net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)
    TEST_DATA_FILE = 'video_list.txt'
    
    height = 227
    width = 227
    flow = False    
    
    f = open(TEST_DATA_FILE, 'r')
    f_lines = f.readlines()
    f.close()

    video_dict = {}
    current_line = 0
    video_order = []
    
    for ix, line in enumerate(f_lines):
      clc_video = line.split(' ')[0].split('/')[0].strip()
            
      frames = glob.glob('%s/*.jpg' %(clc_video))

      num_frames = len(frames)
      video_dict[clc_video] = {}
      video_dict[clc_video]['frames'] = frames[0].split('/frame')[-2] + '/frame_%05d.jpg'
      video_dict[clc_video]['reshape'] = (240,320)
      video_dict[clc_video]['num_frames'] = num_frames
      video_order.append(clc_video) 
      

    video_dict = video_dict
    num_videos = len(video_dict.keys())
    
    #set up data transformer
    
    shape = (1, 3, height, width)
        
    transformer = caffe.io.Transformer({'data_in': shape})
    transformer.set_raw_scale('data_in', 255)
    if flow:
      image_mean = [128, 128, 128]
      transformer.set_is_flow('data_in', True)
    else:
      image_mean = [103.939, 116.779, 128.68]
      transformer.set_is_flow('data_in', False)
    channel_mean = np.zeros((3,227,227))
    for channel_index, mean_val in enumerate(image_mean):
      channel_mean[channel_index, ...] = mean_val
    
    transformer.set_mean('data_in', channel_mean)
    transformer.set_channel_swap('data_in', (2, 1, 0))
   
    transformer.set_transpose('data_in', (2, 0, 1))
    
    idx_list = range(0, num_videos)
    features = [] 
    labels = []
    eval_frames =[]
    for j in idx_list:
        key = video_order[j]
        video_reshape = video_dict[key]['reshape']
        num_frames = video_dict[key]['num_frames']
        frames = video_dict[key]['frames']
        video_frames = []
        video_feat = []
        jump_idx = 0 
        
        for i in range(0,num_frames):#range(np.round(num_frames/20)+1):#Analysis each 20 frames
          idx = i + 1
          if (idx > num_frames):
                idx = num_frames
          curr_frame = frames % idx
          
          
          data_in = caffe.io.load_image(curr_frame)
          
          if (data_in.shape[0] < video_reshape[0]) | (data_in.shape[1] < video_reshape[0]):
            data_in = caffe.io.resize_image(data_in, video_reshape)
          
          processed_image = transformer.preprocess('data_in',data_in)
          processed_image = np.reshape(processed_image, (1,3,227,227))
          out = net.forward_all(blobs=['fc7'],data=processed_image)
          
          features.append(out['fc7'][0])
          video_feat.append(out['fc7'][0])
         
          video_frames.append(curr_frame)
          eval_frames.append(curr_frame)
          
          #labels.append(label)
          print "Frame {}/{}, done".format(idx, num_frames)
        
        print "Video {}: {}, done".format(j, key)
        video_feat = np.vstack(video_feat)
        video_frames = np.hstack(video_frames)
        
        res = dict()
        res['feat'] = video_feat
        res['frames'] = video_frames
        sio.savemat('results/{}'.format(key),res)
    
   
if __name__ == '__main__':
    main()
