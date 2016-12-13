import os
import numpy as np
import matplotlib.pyplot as plt
import math

caffe_root = '/home/jsweet/Documents/Code/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_mode_cpu()
net = caffe.Classifier('deploy.prototxt', 'Sandipan1_Full_26Drugs_iter_50000.caffemodel',
                       mean=np.load('imagenet_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


input_image = caffe.io.load_image(sys.argv[1])
prediction = net.predict([input_image])
pClass = prediction[0].argmax()
print file
print '\tClass:', pClass
print '\tProbability', prediction[0][pClass]
