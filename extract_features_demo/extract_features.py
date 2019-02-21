#!/usr/bin/env python
import numpy as np
import scipy.io as sio
import glob
import caffe


# GLOBALS
SET_MODE = 'gpu' # 'cpu' to use the cpu mode
DEVICE_ID = 0 # Choose your gpu ID, if you are using gpu mode

PRETRAINED_FILE = 'path_to_one_of_our_caffe_models.caffemodel' # IMPORTANT: You need to select a pretrained model
#Below you have an example for the Lq loss trained with dataset UCF101
#PRETRAINED_FILE = '../models/UCF101/lq_loss/caffe_models/Lq_loss_iter_5000.caffemodel' 


MODEL_FILE = 'path_to_the_correspoding_deploy.prototxt'  #IMPORTANT: You need to select the corresponding deploy.prototxt
#Below you have an example for the Lq loss trained with dataset UCF101
#MODEL_FILE = '../src/UCF101/lq_loss/prototxt/deploy_Lq_loss.prototxt' 


TEST_DATA_FILE = 'video_list.txt'
HEIGHT = 227
WIDTH = 227


def main():

    if SET_MODE == 'gpu':
        caffe.set_mode_gpu()
        caffe.set_device(DEVICE_ID)
    elif SET_MODE == 'cpu':
        caffe.set_mode_cpu()

    net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)

    # Make sure the video list file provided does not have a blank line at the end
    with open(TEST_DATA_FILE, 'r') as f:
        f_lines = f.readlines()

    video_dict = {}
    video_order = []
    
    for ix, line in enumerate(f_lines):
        clc_video = line.split(' ')[0].split('/')[0].strip()
        frames = glob.glob('%s/*.jpg' %(clc_video))

        num_frames = len(frames)
        video_dict[clc_video] = {}
        video_dict[clc_video]['frames'] = frames[0].split('/frame')[-2] + '/frame_%05d.jpg'
        video_dict[clc_video]['reshape'] = (240, 320)
        video_dict[clc_video]['num_frames'] = num_frames
        video_order.append(clc_video)

    video_dict = video_dict
    num_videos = len(video_dict.keys())
    
    # Set data transformer up
    shape = (1, 3, HEIGHT, WIDTH)
        
    transformer = caffe.io.Transformer({'data_in': shape})
    transformer.set_raw_scale('data_in', 255)
    image_mean = [103.939, 116.779, 128.68]

    channel_mean = np.zeros((3,227,227))
    for channel_index, mean_val in enumerate(image_mean):
        channel_mean[channel_index, ...] = mean_val
    
    transformer.set_mean('data_in', channel_mean)
    transformer.set_channel_swap('data_in', (2, 1, 0))
    transformer.set_transpose('data_in', (2, 0, 1))
    
    idx_list = range(0, num_videos)
    features = []
    eval_frames =[]
    for j in idx_list:
        key = video_order[j]
        video_reshape = video_dict[key]['reshape']
        num_frames = video_dict[key]['num_frames']
        frames = video_dict[key]['frames']
        video_frames = []
        video_feat = []
        
        for i in range(0,num_frames): # range(np.round(num_frames/20)+1):  --> Analysis each 20 frames
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
