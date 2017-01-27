#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from PIL import Image, ImageEnhance, ImageStat

# function to return average brightness of an image
# Source: http://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python
def brightness(im):
    stat = ImageStat.Stat(im)
    r,g,b = stat.mean
    #return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))   #this is a way of averaging the r g b values to derive "human-visible" brightness
    return math.sqrt(0.577*(r**2) + 0.577*(g**2) + 0.577*(b**2))

#setup drugs list
drugs = ["dog",
         "cat",
         "mouse",
         "camel",
         "horse",
         "wolf",
         "tiger",
         "elephant",
         "badger"
         ]

exclude = "AJ"

image_height = 490

#caffe_root = '/home/jsweet/Documents/Code/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
#sys.path.insert(0, caffe_root + 'python')
#set to my folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import caffe

#get nnet file contents
with open(sys.argv[2]+'.nnet') as fin:
    content = fin.readlines()

fin.close()

#strip spaces
content = [x.strip() for x in content]

#parse contents of definition file
model = ""
imagenet = ""
deploy = ""
target_brightness = 0

for line in content:
    if 'DRUGS' in line:
        drugs = line[6:].split(',')
    elif 'LANES' in line:
        exclude = line[6:]
    elif 'WEIGHTS' in line:
        model = line[8:]
    elif 'IMAGENET' in line:
        imagenet = line[9:]
    elif 'DEPLOY' in line:
        deploy = line[7:]
    elif 'DEPLOY' in line:
        deploy = line[7:]
    elif 'BRIGHTNESS' in line:
        target_brightness = float(line[11:])
    elif 'IMAGEHEIGHT' in line:
        image_height = int(line[12:])

print "Drugs", drugs
print "Exclude", exclude
print "Model", model
print "Imagenet", imagenet
print "Deploy", deploy
print "Brightness", target_brightness
print "Image height", image_height

#get processed image
img = Image.open(sys.argv[1])
    
#brightness, if set. Was set to 165.6 for 4 drug average
if target_brightness > 0:
    bright = brightness(img)
    
    #massage image
    imgbright = ImageEnhance.Brightness(img)
    img = imgbright.enhance(target_brightness/bright)

#open filename /var/www/html/joomla/neuralnetworks/
pos1 = sys.argv[1].rfind('-')
pos2 = sys.argv[1].rfind('.')

randpart = sys.argv[1][pos1+1:pos2-10]
#print "Rand part", randpart

f = open('nnet/nn'+randpart+'.csv',"w+")

#crop comparison
#crop comparison
img = img.crop((71, 359, 71+636, 359+image_height))
    
#lanes split
lane = []

#loop over lanes
for i in range(0,12):
    if chr(65+i) not in exclude:
        lane.append(img.crop((53*i, 0, 53*(i+1), image_height)))

#reconstruct
imgout = Image.new("RGB", (53 * len(lane),image_height))
    
#loop over lanes
for i in range(0,len(lane)):
    imgout.paste(lane[i], (53*i, 0, 53*(i+1), image_height))

#resize and save
imgout = imgout.resize((227,227), Image.ANTIALIAS)
imgout.save('classify/test'+randpart+'.png')

#find prediction
caffe.set_mode_cpu()
net = caffe.Classifier(deploy, model,
                       mean=np.load(imagenet).mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(227, 227))


input_image = caffe.io.load_image('classify/test'+randpart+'.png')
prediction = net.predict([input_image])

#find the highest probability
temppred = copy.deepcopy(prediction[0])
pClass1 = temppred.argmax()

temppred[pClass1] = 0
pClass2 = temppred.argmax()

temppred[pClass2] = 0
pClass3 = temppred.argmax()

#save to temp file
f.write(drugs[pClass1]+','+str(prediction[0][pClass1])+','+str(pClass1)+','+drugs[pClass2]+','+str(prediction[0][pClass2])+','+str(pClass2)+','+drugs[pClass3]+','+str(prediction[0][pClass3])+','+str(pClass3)+',\r\n')

f.close()

#print results
print '\tClass:', pClass1, pClass2, pClass3
print '\tProbability', prediction[0][pClass1], prediction[0][pClass2], prediction[0][pClass3]
print '\tDrug', drugs[pClass1], drugs[pClass2], drugs[pClass3]

#exit
sys.exit(0)
