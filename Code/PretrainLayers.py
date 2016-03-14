## pretraining for DBM with softmax visible units
#
# this script does the following:
#	 - first first layer (softmax visible units) we divide data into visible and hidden units and select architecture (ie number of hidden units) based on log likelihood on held out data
#	 - this requires us to estimate the partition functions for each RBM
#	 - we select architecture with best log-likelihood (50 hidden units) and repeat with top layer
#
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

# load in data:
os.chdir(DataDir)
Dat = pickle.load(open("WordIncidence_1000.p", "rb"))
Xword = pickle.load(open("Words_1000.p", "rb"))

# all preprocessing has been already done. It was as follows:
#   - remove stopwords (using nltk.stopwors)
#   - remove words occuring too frequently (over 125000 over all docs) and not frequently enough (less than 50 across all docs) 
#   - removed all words of length 2 (such "an", "ie", "et", etc)
#   - selected 1000 remaining words which occured most frequently (this was approx 82% of all occurances)
#
#

# train for a range of hidden units - then use validation data to select architecture based on log-likelihood:
h_units = [20,30,40,50,75] 
for h in h_units:
	l1RBM = softmaxRBM(data= Dat[ :9574, :], validation_data= Dat[ 9574:,:], wordVec = Xword)
	l1RBM.train(h_units=h, epochs=5000, batch_size=100, lr=0.0001, momentum=0.9, dropOutProb=.9, numpyRNG=None, gradClip=-1, layer="bottom", verbose=True)
	os.chdir("/media/ricardo/TheBadBoy/Neurosynth/Text/Deep Models/Weights")
	pickle.dump(l1RBM, open("firstLayer_h" +str(h) + "_preTrained.p", "wb"))

# estimate partition functions and likelihood over validation data:
os.chdir(StoreDir)
ValidationLikelihood = {}
for h in h_units:
	l1RBM = pickle.load(open("firstLayer_h" +str(h) + "_preTrained.p", "rb"))
	l1RBM.estimateZ_AIS(steps=1000, M=500) # might not be enough steps...
	ValidationLikelihood[h] = []
	for x in xrange(l1RBM.val_data.shape[0]):
		if l1RBM.val_data[x].sum()>0:
			ValidationLikelihood[h].append( l1RBM.FreeEnergy(l1RBM.val_data[x], partition=True)  )

h_mean = [numpy.mean(ValidationLikelihood[h]) for h in h_units]
h_std = [numpy.std(ValidationLikelihood[h]) for h in h_units]

data = [ValidationLikelihood[h] for h in h_units]
plt.boxplot(data)

## based on these results we selct 50 hidden units!
h = 50
l1RBM = pickle.load(open("firstLayer_h" +str(h) + "_preTrained.p", "rb"))
# now train top layer RBM:
trainSamp = 50000
data_l2 = l1RBM.generateHiddenSamples(n=trainSamp)


h_units = [20,30,40,50,75,100]
for h in h_units:
	l2RBM = dropoutRBM(data=data_l2[: int(trainSamp*.9)], validation_data=data_l2[int(trainSamp*.9):]) # layer 2 RBM
	l2RBM.train(h_units = h, epochs=5000, batch_size=150, lr=0.00001, momentum=0.9, dropOutProb=.9, GibbsSteps=1, numpyRNG=None, layer="top")
	os.chdir(StoreDir)
	pickle.dump(l2RBM, open("secondLayer_h" +str(h) + "_preTrained.p", "wb"))

ValidationLikelihoodL2 = {}
for h in h_units:
	l2RBM = pickle.load(open("secondLayer_h" +str(h) + "_preTrained.p", "rb"))
	l2RBM.estimateZ_AIS(steps=1000, M=500) # might not be enough steps...
	ValidationLikelihoodL2[h] = []
	for x in xrange(l2RBM.val_data.shape[0]):
		if l2RBM.val_data[x].sum()>0:
			ValidationLikelihoodL2[h].append( l2RBM.FreeEnergy(l2RBM.val_data[x], partition=True)  )


data = [ValidationLikelihoodL2[h] for h in h_units]
plt.boxplot(data)

