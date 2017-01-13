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
from PIL import Image, ImageEnhance, ImageStat
import copy
import math

# function to return average brightness of an image
# Source: http://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python
def brightness(im):
    stat = ImageStat.Stat(im)
    r,g,b = stat.mean
    #return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))   #this is a way of averaging the r g b values to derive "human-visible" brightness
    return math.sqrt(0.577*(r**2) + 0.577*(g**2) + 0.577*(b**2))

#inline parameter?
optlist, args = getopt.getopt(sys.argv[1:], 'i:c:f:t:s:m:d:e:n:b:')

#setup drugs list
drugs = ["Amoxicillin",
         "Acetaminophen",
         "Ciprofloxacin",
         "Ceftriaxone"
         ]

exclude = ""
sampleID = ""
catagory = ""
test = ""
sample = ""
model = 'pad_aug1_110000.caffemodel'
outfilename = "tmp/drugs.csv"
imagenet = 'imagenet_mean.npy'
image_height = 490
#was 490 to include reactions below test line
target_brightness = 0

for o, a in optlist:
    if o == '-i':
        sampleID = a
        print "Sample ID", sampleID
    elif o == '-c':
        catagory = a
        print "Catagory", catagory
    elif o == '-f':
        outfilename = a
        print "File name", outfilename
    elif o == '-t':
        test = a
        print "Test name", test
    elif o == '-s':
        sample = a
        print "Sample name", sample
    elif o == '-m':
        model = a
        print "Caffe model", model
    elif o == '-d':
        drugs = a.split(',')
        print "Drugs", ', '.join(drugs)
    elif o == '-e':
        exclude = a
        print "Exclusions", exclude
    elif o == '-n':
        imagenet = a
        print "Imagenet numpy", imagenet
    elif o == '-b':
        target_brightness = float(a)
        print "Target brightness", target_brightness
    else:
        print 'Unhandled option: ', o
        sys.exit(-2)

positive_classify = [0] * len(drugs)
negative_classify = [0] * len(drugs)

#set to my folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#parse ID
sid = 0
if sampleID != "":
    sid = int(sampleID)

#check input data
if sid == 0 and catagory == "" and test == "" and sample == "":
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
sqlcommand = 'SELECT `id`,`processed_file_location`,`sample_name`,`sample_id` FROM `card` WHERE `processed_file_location`!=""'

if sid != 0:
    sqlcommand = sqlcommand + ' AND `sample_id`='+str(sid)
    #cur.execute('SELECT `id`,`processed_file_location`,`sample_name`,`sample_id` FROM `card` WHERE `sample_id`='+str(sid)+' AND `processed_file_location`!=""')

if test != "":
    sqlcommand = sqlcommand + ' AND `test_name`="'+test+'"'
    #cur.execute('SELECT `id`,`processed_file_location`,`sample_name`,`sample_id` FROM `card` WHERE `test_name`="'+test+'" AND `processed_file_location`!=""')

if sample != "":
    sqlcommand = sqlcommand + ' AND `sample_name`="'+sample+'"'
    #cur.execute('SELECT `id`,`processed_file_location`,`sample_name`,`sample_id` FROM `card` WHERE `sample_name`="'+sample+'" AND `processed_file_location`!=""')

if catagory != "":
    sqlcommand = sqlcommand + ' AND `category`="'+catagory+'"'
    #cur.execute('SELECT `id`,`processed_file_location`,`sample_name`,`sample_id` FROM `card` WHERE `category`="'+catagory+'" AND `processed_file_location`!=""')

#print and execute
print "SQL command", sqlcommand
cur.execute(sqlcommand)

#open file?
if outfilename != "":
    f = open(outfilename,"w+")

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

    #brightness, if set. Was set to 165.6 for 4 drug average
    if target_brightness > 0:
        bright = brightness(img)
    
        #massage image
        imgbright = ImageEnhance.Brightness(img)
        img = imgbright.enhance(target_brightness/bright)

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
    imgout.save('tmp/test.png')

    #predict
    caffe.set_mode_cpu()
    net = caffe.Classifier('deploy.prototxt', model,
                           mean=np.load(imagenet).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(227, 227))


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
for i in range(0,len(drugs)):
    temptotal = positive_classify[i] + negative_classify[i]
    if temptotal > 0:
        print "Drug",drugs[i], positive_classify[i], temptotal
        if f:
            f.write(drugs[i]+','+str(positive_classify[i])+','+str(temptotal)+',\r\n')

if f:
    f.close()

# Close all cursors
cur.close()
# Close all databases
db.close()
