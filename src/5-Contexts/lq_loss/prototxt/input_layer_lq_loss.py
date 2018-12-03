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

RGB_frames = '../../../../personal_contexts_dataset'
test_frames = 1 
train_frames = 1
pair = 23
test_buffer = 3
train_buffer = 3#24
CAMERA = 'ALL'


 
def BatchAdvancer(idx, buffer_size):
    idx += buffer_size
    return idx
  
class videoRead(caffe.Layer):

  def initialize(self):
    self.data = []
    self.videos = []
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
    self.path_to_images = RGB_frames + '/' + CAMERA + '/' + 'Test'
    self.video_list = 'videos_all_rand.txt' 

  def setup(self, bottom, top):
    random.seed(10)
    self.initialize()
    f = open(self.video_list, 'r')
    f_lines = f.readlines()
    f.close()

    current_line = 0
    for ix, line in enumerate(f_lines):
      video =  line.split(' ')[0]
      
      self.videos.append(video) 
      
      

    
    self.num_videos = len(self.videos)
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
        key = self.videos[j]
        im_folder = self.path_to_images + '/' + key.strip() +'/*.jpg'
          
        frames = glob.glob(im_folder)
        num_frames = len(frames)
       
        im_flip = []
        jump_idx = 0
        if (num_frames < 100):
            jump_idx2 = 7
        else:
            jump_idx2 = 17
        index = 0
        for i in range(pair):
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
          
          if (idx2 > num_frames-1):
               idx2 = num_frames-1
          
          if (idx3 > num_frames):
                idx3 = num_frames
          
          curr_frame = self.path_to_images + '/' + key.strip() + '/' + str(idx) + '.jpg'
          data_in = caffe.io.load_image(curr_frame)
          processed_image = self.transformer.preprocess('data_in',data_in)
          #plt.imshow( processed_image.transpose(1, 2, 0).astype(np.uint8))
          #plt.show()
          
          #consecutive frame
          next_frame = self.path_to_images + '/' + key.strip() + '/' + str(idx2) + '.jpg'
          data_in = caffe.io.load_image(next_frame)
          
          processed_image2 = self.transformer.preprocess('data_in',data_in)

          #same video frame
          video_frame = self.path_to_images + '/' + key.strip() + '/' + str(idx3) + '.jpg'
          data_in = caffe.io.load_image(video_frame)
          
          processed_image3 = self.transformer.preprocess('data_in',data_in)
          
          #negative frame --> Semi-supervised!!!!
          v_idx = anchor_idx
          while (v_idx == anchor_idx):
               v_idx = random.randint(0, len(idx_list)-1)
               key2 = self.videos[v_idx]
                       
          frames2 = glob.glob(self.path_to_images + '/' + key2.strip() + '/*.jpg')
          num_frames2 = len(frames2)
            
          i_idx = random.randint(1, num_frames2)
          neg_frame = self.path_to_images + '/' + key2.strip() + '/' + str(i_idx) + '.jpg'
            
          data_in = caffe.io.load_image(neg_frame)
          
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
    self.videos = []
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
    self.path_to_images =  RGB_frames + '/' + CAMERA + '/' + 'Training'
    self.video_list = 'videos_all_rand.txt' 

class videoReadTest_RGB(videoRead):

  def initialize(self):
    self.data = []
    self.videos = []
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
    self.path_to_images = RGB_frames + '/' + CAMERA + '/' + 'Test'
    self.video_list = 'videos_all_rand.txt' 
