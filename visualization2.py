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

largest_class = 0
expected_class = {}
class_name = []
for line in [line.rstrip('\r\n') for line in open('Sandipan1_Full_test1.txt')]:
    parts = line.split(" ")
    if len(parts) == 2:
        expected_class[parts[0]] = int(parts[1])
        largest_class = max(largest_class, int(parts[1]))

        name = parts[0].split("_")[0]
        if not name in class_name:
            class_name.append(name)

var = os.listdir('Sandipan1_Full_test1')
var = sorted(var)

statdata = []

classPassData = []
classTotalData = []
classConfData = []
for i in range(largest_class + 1):
    classPassData.append(0)
    classTotalData.append(0)
    classConfData.append([])
    
Drug = 0
Pred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
f1 = open('Prediction1.csv','w')

for file in var:
    if file.endswith('.jpg'):
        input_image = caffe.io.load_image('Sandipan1_Full_test1/' + file)
        prediction = net.predict([input_image])
        #print len(prediction[0])
        file1 = file.split('.')
        file2 = file1[0].split('_')
        if Drug == 0:
            Drug = file2[0]
        elif Drug != file2[0]:
            #f1.write(Drug)
            f1.write(' ')
            for i in range(0,len(Pred)):
                f1.write(str(Pred[i]/10))
                f1.write(' ')
            f1.write('\n')
            Drug = file2[0]
            Pred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for j in range(0,26):
            Pred[j] = Pred[j] + prediction[0][j]
        print Drug
        pClass = prediction[0].argmax()
        print file
        print '\tClass:', pClass
        print '\tProbability', prediction[0][pClass]

        classTotalData[expected_class[file]] += 1
        if prediction[0].argmax() == expected_class[file]:
            print '\tPredicted Correctly'
            classPassData[expected_class[file]] += 1

        statdata.append(prediction[0][expected_class[file]])
        classConfData[expected_class[file]].append(prediction[0][expected_class[file]])
        
#f1.write(Drug)
f1.write(' ')
for i in range(0,len(Pred)):
    f1.write(str(Pred[i]/10))
    f1.write(' ')
f1.write('\n')
f1.close()

print np.average(statdata)
for i in range(len(classConfData)):
    print "Class:", class_name[i], "[", classPassData[i], "/", classTotalData[i], "] -", np.average(classConfData[i])
