### DBM fine tuning
#
# this script applies fine tuning to the pretrained RBM layers
#
#

import os
import csv
import cPickle as pickle
import numpy
import pylab as plt

CodeDir = "../Code" # directory containing code (i.e. softmaxRBM.py and dropoutRBM.py files)
DataDir = "../Data" # directory containing data
StoreDir = "../Weights" # directory to store results

os.chdir(CodeDir)
from softmaxRBM import *
from dropoutRBM import *
from DBM import *

# load in pretrained layers:
os.chdir(StoreDir)
l1RBM = pickle.load(open("firstLayer_h50_preTrained.p", "rb"))
l2RBM = pickle.load(open("secondLayer_h50_preTrained.p", "rb"))

dbm = softmaxDBM_3layer(softmaxLayer=l1RBM, RBMlayer=l2RBM)

# now start with the formal training of DBM:
dbm.train(epochs=15000, M=500, batch_size=500, lr=0.00001, momentum=0.5, decRate=.999, verbose=True)

os.chdir(StoreDir)
pickle.dump(dbm, open("DBM_"+str(dbm.softmaxLayer.h_units) + "_" + str(dbm.l2layer.h_units) + "_Full.p", "wb"))



