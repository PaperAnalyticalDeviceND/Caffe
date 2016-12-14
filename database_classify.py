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

positive_classify = [0] * 26
negative_classify = [0] * 26

#inline parameter?
optlist, args = getopt.getopt(sys.argv[1:], 'i:c:f:')

sampleID = ""
catagory = ""
filename = "tmp/drugs.csv"

for o, a in optlist:
    if o == '-i':
        sampleID = a
        print "Sample ID", sampleID
    elif o == '-c':
        catagory = a
        print "Catagory", catagory
    elif o == '-f':
        filename = a
        print "File name", filename
    else:
        print 'Unhandled option: ', o
        sys.exit(-2)

#set to my folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#parse ID
sid = 0
if sampleID != "":
    sid = int(sampleID)

#check input data
if sid == 0 and catagory == "":
    print "Insufficient input data"
    sys.exit(-1)

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
if sid != 0:
    cur.execute('SELECT `id`,`processed_file_location`,`sample_name`,`sample_id` FROM `card` WHERE `sample_id`='+str(sid))
else:
    cur.execute('SELECT `id`,`processed_file_location`,`sample_name`,`sample_id` FROM `card` WHERE `catagory`='+catagory)

#open file?
if filename != "":
    f = open(filename,"w+")

# print all the first cell of all the rows
for row in cur.fetchall() :
    print row[0],row[1]
    
    #test that drug exists
    if row[2] not in drugs:
        print "Drug ", row[2], "not in training list!"
        #sys.exit(-2)
        continue
    
    #get drug index
    drugindex = drugs.index(row[2])

    #get input data
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
    net = caffe.Classifier('deploy.prototxt', 'Sandipan1_Full_26Drugs_iter_50000.caffemodel',
                           mean=np.load('imagenet_mean.npy').mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))


    input_image = caffe.io.load_image('tmp/test.png')
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
    print '\tExpected', row[2]

    os.remove('tmp/test.png')

    #update stats
    if drugindex == pClass1:
        positive_classify[drugindex] = positive_classify[drugindex] + 1
    else:
        negative_classify[drugindex] = negative_classify[drugindex] + 1

    #save data
    if f:
        f.write(str(row[0])+','+str(row[3])+','+str(row[2])+','+drugs[pClass1]+','+str(prediction[0][pClass1])+','+str(pClass1)+','+drugs[pClass2]+','+str(prediction[0][pClass2])+','+str(pClass2)+','+drugs[pClass3]+','+str(prediction[0][pClass3])+','+str(pClass3)+',\r\n')


    #just do first instance
    #break

#print stats
for i in range(0,26):
    temptotal = positive_classify[i] + negative_classify[i]
    if temptotal > 0:
        print "Drug",drugs[i], positive_classify[i] / temptotal, temptotal
        if f:
            f.write(drugs[i]+','+str(positive_classify[i] / temptotal)+','+str(temptotal)+',\r\n')

if f:
    f.close()

# Close all cursors
cur.close()
# Close all databases
db.close()
