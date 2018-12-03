#!/usr/bin/env python

#Data layer for video.  Change flow_frames and RGB_frames to be the path to the flow and RGB frames.

import sys
sys.path.append('../../python')
import caffe
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import time
import pdb
import glob
import pickle as pkl
import random
import h5py
from multiprocessing import Pool
from threading import Thread
import skimage.io
import copy

flow_frames = 'flow_images/'
RGB_frames = 'RGB/'
test_frames = 1 
train_frames = 1
pair = 15
test_buffer = 3
train_buffer = 3#24


 
def BatchAdvancer(idx, buffer_size):
    idx += buffer_size
    return idx
  
class videoRead(caffe.Layer):

  def initialize(self):
    self.data = []
    self.Ndata = 0
    self.train_or_test = 'test'
    self.flow = False
    self.buffer_size = test_buffer  #num videos processed per batch
    self.frames = test_frames   #length of processed clip
    self.N = self.buffer_size*pair
    self.idx = 0
    self.channels = 12
    self.height = 227
    self.width = 227
    self.path_to_images = RGB_frames 
    self.video_list = 'ucf101_split1_testVideos_check2.txt' 

  def setup(self, bottom, top):
    random.seed(10)
    self.initialize()
    f = open(self.video_list, 'r')
    f_lines = f.readlines()
    f.close()

    video_dict = {}
    current_line = 0
    self.video_order = []
    for ix, line in enumerate(f_lines):
      clc_video = line.split(' ')[0].split('/')[0]
      video = line.split(' ')[0].split('/')[1]
      l = int(line.split(' ')[1])
      frames = glob.glob('%s%s/%s/*.jpg' %(self.path_to_images, clc_video, video))
      num_frames = len(frames)
      video_dict[video] = {}
      video_dict[video]['frames'] = frames[0].split('.')[0] + '.%04d.jpg'
      video_dict[video]['reshape'] = (240,320)
      video_dict[video]['num_frames'] = num_frames
      video_dict[video]['label'] = l
      self.video_order.append(video) 
      

    self.video_dict = video_dict
    self.num_videos = len(video_dict.keys())
    #set up data transformer
    
    shape = (self.N, self.channels-9, self.height, self.width)
        
    self.transformer = caffe.io.Transformer({'data_in': shape})
    self.transformer.set_raw_scale('data_in', 255)
    if self.flow:
      image_mean = [128, 128, 128]
      self.transformer.set_is_flow('data_in', True)
    else:
      image_mean = [103.939, 116.779, 128.68]
      self.transformer.set_is_flow('data_in', False)
    channel_mean = np.zeros((3,227,227))
    for channel_index, mean_val in enumerate(image_mean):
      channel_mean[channel_index, ...] = mean_val
    self.transformer.set_mean('data_in', channel_mean)
    self.transformer.set_channel_swap('data_in', (2, 1, 0))
   
    self.transformer.set_transpose('data_in', (2, 0, 1))    

  
  
  def reshape(self, bottom, top):
      top[0].reshape(self.N, *(self.channels, 227, 227))#self.data[0].shape
      #top[1].reshape(self.N)
      #print "self.data[0].shape {}".format(top[0])
      #raw_input()   
  

  def forward(self, bottom, top):
  
      data_aux = []
      data = []
      self.Ndata = 0
      
      if self.idx + self.buffer_size >= self.num_videos:
        if (self.idx-self.num_videos < 0):
            idx_list = range(self.idx, self.num_videos)
            idx_list.extend(range(0, self.buffer_size-(self.num_videos-self.idx)))
        else:
            [dumm, idx_aux] = divmod(self.idx,self.num_videos)
            if idx_aux+self.buffer_size <= self.num_videos:
                idx_list = range(idx_aux, idx_aux+self.buffer_size)
            else:
                idx_list = range(idx_aux, self.num_videos)
                idx_list.extend(range(0, self.buffer_size-(self.num_videos-idx_aux)))
        
      else:
         
         idx_list = range(self.idx, self.idx+self.buffer_size)
            
      
      
      for j in idx_list:
        anchor_idx = j
        key = self.video_order[j]
        video_reshape = self.video_dict[key]['reshape']
        num_frames = self.video_dict[key]['num_frames']
        frames = self.video_dict[key]['frames']
       
        im_flip = []
        jump_idx = 0
        if (num_frames < 100):
            jump_idx2 = 7
        else:
            jump_idx2 = 17
        index = random.randint(0, num_frames)#0
        flag = 0;
        for i in range(pair):
          if flag:
             index = random.randint(0, num_frames)
             jump_idx = 0
             flag = 0
             
          idx = jump_idx + (index+1)
          idx2 = idx + 1
          idx3 = idx + jump_idx2
         
          index += 1
          if (num_frames < 100):
            jump_idx += 2
          else:
            jump_idx += 7
          
          if (idx > num_frames-2):
                idx = num_frames-2
                flag = 1
          
          if (idx2 > num_frames-1):
               idx2 = num_frames-1
          
          if (idx3 > num_frames):
                idx3 = num_frames
          
          curr_frame = frames % idx
          f = random.randint(0,1)
          im_flip.extend([f]) 
          data_in = caffe.io.load_image(curr_frame)
          if (data_in.shape[0] < video_reshape[0]) | (data_in.shape[1] < video_reshape[0]):
            data_in = caffe.io.resize_image(data_in, video_reshape)
          #if im_flip:
          #  data_in = caffe.io.flip_image(data_in, 1, self.flow) 
         
          processed_image = self.transformer.preprocess('data_in',data_in)
          #plt.imshow( processed_image.transpose(1, 2, 0).astype(np.uint8))
          #plt.show()
          
          #consecutive frame
          next_frame = frames % idx2
          f = random.randint(0,1)
          im_flip.extend([f]) 
          data_in = caffe.io.load_image(next_frame)
          if (data_in.shape[0] < video_reshape[0]) | (data_in.shape[1] < video_reshape[0]):
               data_in = caffe.io.resize_image(data_in, video_reshape)
          #if im_flip:
          #     data_in = caffe.io.flip_image(data_in, 1, self.flow) 
         
          processed_image2 = self.transformer.preprocess('data_in',data_in)

          #same video frame
          video_frame = frames % idx3
          f = random.randint(0,1)
          im_flip.extend([f]) 
          data_in = caffe.io.load_image(video_frame)
          if (data_in.shape[0] < video_reshape[0]) | (data_in.shape[1] < video_reshape[0]):
               data_in = caffe.io.resize_image(data_in, video_reshape)
          #if im_flip:
          #     data_in = caffe.io.flip_image(data_in, 1, self.flow) 
         
          processed_image3 = self.transformer.preprocess('data_in',data_in)
          
          #negative frame --> Semi-supervised!!!!
          v_idx = anchor_idx
          while (v_idx == anchor_idx):
             v_idx = random.randint(0, len(idx_list)-1)
             key2 = self.video_order[v_idx]
             
            
          num_frames2 = self.video_dict[key2]['num_frames']
          frames2 = self.video_dict[key2]['frames']
            
          i_idx = random.randint(1, num_frames2)
          neg_frame = frames2 % i_idx
            
          f = random.randint(0,1)
          im_flip.extend([f]) 
          data_in = caffe.io.load_image(neg_frame)
          if (data_in.shape[0] < video_reshape[0]) | (data_in.shape[1] < video_reshape[0]):
               data_in = caffe.io.resize_image(data_in, video_reshape)
          #if im_flip:
          #     data_in = caffe.io.flip_image(data_in, 1, self.flow) 
         
          processed_image4 = self.transformer.preprocess('data_in',data_in)

          data_aux.append(np.vstack((processed_image, processed_image2, processed_image3, processed_image4)))
          #plt.imshow( processed_image2.transpose(1, 2, 0).astype(np.uint8))
          #plt.show()
          #plt.imshow( processed_image.transpose(1, 2, 0).astype(np.uint8))
          #plt.show()
                   
          self.Ndata += 1
          
      #print "self.Ndata2 {}".format(self.Ndata)
      #raw_input()
          
      if (self.Ndata > self.N):
          for i in range(self.N):
              data.append(data_aux[i])
              
      if (self.Ndata == self.N):
           data = data_aux
    
                    
      #print "Datos {}".format(len(self.label)) 
      #print "Train/Test {}".format(self.train_or_test)    
      self.idx = BatchAdvancer(self.idx, self.buffer_size)
      #print "idx {}".format(self.idx)     
      #raw_input()
      # assign output
      
      for k in range(self.N):
            top[0].data[k,...] = data[k]
        
  
  def backward(self, top, propagate_down, bottom):
      pass

class videoReadTrain_RGB(videoRead):
  
  def initialize(self):
    self.data = []
    self.Ndata = 0
    self.train_or_test = 'train'
    self.flow = False
    self.buffer_size = train_buffer  #num videos processed per batch
    self.frames = train_frames   #length of processed clip
    self.N = self.buffer_size*pair
    self.idx = 0
    self.channels = 12
    self.height = 227
    self.width = 227
    self.path_to_images = RGB_frames 
    self.video_list = 'ucf101_split1_trainVideos.txt' 

class videoReadTest_RGB(videoRead):

  def initialize(self):
    self.data = []
    self.Ndata = 0
    self.train_or_test = 'test'
    self.flow = False
    self.buffer_size = test_buffer  #num videos processed per batch
    self.frames = test_frames   #length of processed clip
    self.N = self.buffer_size*pair
    self.idx = 0
    self.channels = 12
    self.height = 227
    self.width = 227
    self.path_to_images = RGB_frames 
    self.video_list = 'ucf101_split1_testVideos_check2.txt' 
