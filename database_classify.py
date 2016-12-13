#!/usr/bin/python
import datetime, os
import sys
import subprocess
import MySQLdb
import getopt
import os
import numpy as np
#import matplotlib.pyplot as plt
import math
import caffe
from PIL import Image

#inline parameter?
optlist, args = getopt.getopt(sys.argv[1:], 'i:')

sampleID = ""

for o, a in optlist:
    if o == '-i':
        sampleID = a
        print "Sample ID", sampleID
    else:
        print 'Unhandled option: ', o
        sys.exit(-2)

#set to my folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#parse ID
sid = int(sampleID)

#get database credentials
with open('credentials.txt') as f:
    line = f.readline()
    split = line.split(",")
f.close()

#open database
db = MySQLdb.connect(host="localhost", # your host, usually localhost
                     user=split[0], # your username
                      passwd=split[1], # your password
                      db="pad") # name of the data base

# you must create a Cursor object. It will let
#  you execute all the queries you need
cur = db.cursor()

# Use all the SQL you like
cur.execute('SELECT `id`,`processed_file_location` FROM `card` WHERE `sample_id`='+str(sid))

# print all the first cell of all the rows
for row in cur.fetchall() :
    print row[0],row[1]
    filename = row[1]
    id = row[0]
    
    if filename == "":
        print "No processed file for ID",str(sid)
        break

    #get processed image
    img = Image.open(filename)

    #crop comparison
    img = img.crop((72, 359, 72+636, 359+490))
    img.save('tmp/test.png')

    #predict
    caffe.set_mode_cpu()
    net = caffe.Classifier('deploy.prototxt', 'Sandipan1_Full_26Drugs_iter_100000.caffemodel',
                           mean=np.load('imagenet_mean.npy').mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))


    input_image = caffe.io.load_image('tmp/test.png')
    prediction = net.predict([input_image])
    pClass = prediction[0].argmax()
    print file
    print '\tClass:', pClass
    print '\tProbability', prediction[0][pClass]
    

    #os.remove('tmp/test.png')

    #just do first instance
    break

# Close all cursors
cur.close()
# Close all databases
db.close()
