import os
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

#setup drugs list
drugs = ["Acetaminophen",
         "AcetylsalicylicAcid",
         "Amodiaquine",
         "Amoxicillin",
         "Ampicillin",
         "Artesunate",
         "CalciumCarbonate",
         "CornStarch",
         "Diethylcarbamazine",
         "Ethambutol",
         "Isoniazid",
         "Rifampicin",
         "Tetracycline",
         "Azithromycin",
         "Chloramphenicol",
         "Chloroquine",
         "Ciprofloxacin",
         "DIWater",
         "DriedWheatStarch",
         "PenicillinG",
         "PotatoStarch",
         "Primaquine",
         "Quinine",
         "Streptomycin",
         "Sulfadoxine",
         "Talc"
         ]

#caffe_root = '/home/jsweet/Documents/Code/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
#sys.path.insert(0, caffe_root + 'python')
#set to my folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import caffe

caffe.set_mode_cpu()
net = caffe.Classifier('deploy.prototxt', 'Sandipan1_Full_26Drugs_iter_50000.caffemodel',
                       mean=np.load('imagenet_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


input_image = caffe.io.load_image(sys.argv[1])
prediction = net.predict([input_image])

temppred = copy.deepcopy(prediction[0])
pClass1 = temppred.argmax()

temppred[pClass1] = 0
pClass2 = temppred.argmax()

temppred[pClass2] = 0
pClass3 = temppred.argmax()

print file
print '\tClass:', pClass1, pClass2, pClass3
print '\tProbability', prediction[0][pClass1], prediction[0][pClass2], prediction[0][pClass3]
print '\tDrug', drugs[pClass1], drugs[pClass2], drugs[pClass3]
