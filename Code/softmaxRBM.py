### implementation of REPLICATED SOFTMAX RESTRICTED BOLTZMANN MACHINE
#
# the following code performed CD-1 to estimate gradients
#

import numpy, scipy, os
import cPickle as pickle
import scipy # needed from logsumexp function!
from scipy.misc import logsumexp 

def getCrossEntropy(visible, PDF):
	"""
	Calculate cross-entropy for observed documents

	"""

	ndocs = visible.shape[0]
	CrossEnt = numpy.zeros((ndocs,))
	for doc in xrange(ndocs):
		CrossEnt[doc] = numpy.dot(visible[doc], numpy.log(PDF[doc]))

	return CrossEnt.mean()

def sigmoid(x):
    """
    sigmoid activation function
    """
    return (1 + scipy.tanh(x/2))/2


class softmaxRBM(object):
	"""

	class for replicated softmax RBM 

	"""

	def __init__(self, data, validation_data, wordVec=None):
		"""
		initialize object

		input:
			 - data: incidence matrix - each row a document, each column the number of times a word appears in that document 
			 - validation_data: same format as data, used to compute validation validation performance 
			 - wordVec: list of words (used for 1 step reconstruction etc)

		"""

		self.data = data
		self.val_data = validation_data
		self.dictsize = data.shape[1] # number of columns indicates number of words
		self.wordVec = wordVec
		self.logZ = None # log parition function

	def generateHiddenSamples(self,  n=1e5, factor=1, norm=False):
		"""
		Generate samples from hidden units. This is crucial for training of DBMs as these samples are used 
		to train further RBMs in the stack

		n is the number of samples to generate (should be a big number!)
		batchsize is number of docs to use when generating each sample (after thinking about it I think this should be 1)
		factor is used when pretraining DBM stack (will have to set to 2)
		norm indicates if we should normalize the scaling coefficients

		"""

		hiddenSamples = numpy.zeros((int(n), self.h_units))# each row will be a sample from hidden units
		ii = numpy.random.randint(low=0, high=self.data.shape[0]-1, size=n )
		visible = self.data[ii,:]
		scaling = visible.sum(axis=1)
		if norm:
			scaling /=  scaling
		hprob = sigmoid( numpy.dot(visible, self.W*factor)*self.bottomUp + numpy.outer(scaling, self.hbias))
		hiddenSamples = (hprob > numpy.random.rand( n,  self.h_units) )*1 

		return hiddenSamples

	def oneStepRecon(self, word, n=10, adjustForDropOut=False):
		"""
		Get a one step reconstruction of a given word

		M is the length of average document (Dat.sum(axis=1).mean())
		n is number of top words to return
		adjustForDropOut: should we correct for dropout (will be required during training)

		"""
		ii = list(self.wordVec).index(word) # word index
		if adjustForDropOut:
			hprob = sigmoid(self.W[ii,:]*self.bottomUp*self.dropOutProb + self.hbias*self.dropOutProb )
			vprob = (  numpy.dot(self.W, hprob)*self.dropOutProb + self.vbias*self.dropOutProb)#+ my_rsm.vbias)

		else:
			hprob = sigmoid(self.W[ii,:]*self.bottomUp + self.hbias )
			vprob = (  numpy.dot(self.W, hprob) + self.vbias)#+ my_rsm.vbias)

		return self.wordVec[vprob.argsort()[::-1][:n]]

	def train(self, h_units, epochs, batch_size, lr, momentum, dropOutProb=.5, numpyRNG=None, gradClip=5, layer="bottom", verbose=False):
		"""
		Train replcated softmax RBM using CD-1
		
		input:
			 - h_units: number of hidden units to employ
			 - epochs: number of training epochs 
			 - batch_size: number of documents in each batch update
			 - lr: learning rate 
			 - momentum: momentum rate (set to 0 for standard GD)
			 - dropOutProb: probability of hidden units being present (ie 1-prob of dropping)
			 - gradClip: value at which we clip the gradient
			 - layer: if layer=="bottom" we assume this training is taking place as the bottom layer in a DBM and adjust weights accordingly
			 - verbose: boolean indicating if we should print out progress

		"""
		# define corresponding variables for weights, etc 
		# these are defined here so we can quickly train with multiple numbers of hidden units
		self.h_units = h_units
		self.lr = lr
		self.momentum = momentum
		self.dropOutProb = dropOutProb
		self.weightsAdjusted = False # indicates if weights have been adjusted (to correct for dropout)

		if layer=="bottom":
			print "training for softmax visible units in DBM... weights will be adjusted accordingly"
			self.bottomUp = 2
		else:
			print "training vanilla softmax RBM"
			self.bottomUp = 1

		if numpyRNG is None:
			self.numpy_rng = numpy.random.RandomState(1234)
		else:
			self.numpy_rng = numpyRNG

		# initialize weights:
		self.W = self.numpy_rng.uniform( low=-4 * numpy.sqrt(6. / (self.h_units + self.dictsize)),
					  high=4 * numpy.sqrt(6. / (self.h_units + self.dictsize)),
					  size=(self.dictsize, self.h_units))
		self.hbias = numpy.zeros((self.h_units, ))
		self.vbias = numpy.zeros((self.dictsize,))

		self.deltaW = numpy.zeros(( self.dictsize, self.h_units ))
		self.deltahbias = numpy.zeros((self.h_units, ))
		self.deltavbias = numpy.zeros((self.dictsize, ))

		# prepare data:
		batch_num = int(numpy.floor(self.data.shape[0] / batch_size))

		# split documents into batches:
		ii = range(self.data.shape[0])
		batchID = [ii[i::batch_num] for i in range(batch_num)]

		print "training replicated softmax RBM..."
		# now we iterative of epochs and over batches:
		for e in xrange(epochs):
			print "running epoch %s" %(e+1)
			for b in xrange(batch_num):
				visible = self.data[batchID[b]] # select visible units
				scaling = visible.sum(axis=1) # scaling required for hidden variables

				# sample dropped units (redo for each batch!):
				r = ( self.dropOutProb > numpy.random.rand( self.h_units ))*1

				## positive phase:
				hidden_f1 = sigmoid( numpy.dot(visible, self.W)*self.bottomUp + numpy.outer(scaling, self.hbias)) # fantasy 1, probability
				hidden_f1_sample = (hidden_f1 > numpy.random.rand( len(batchID[b]),  self.h_units) )*1 # fantasy 1, sample - needed for CD
				hidden_f1_sample *= r # remove dropped units

				## negative phase (no loop as we are performing CD-1!):
				visible_f1 = numpy.exp(numpy.dot(self.W, hidden_f1_sample.T) + self.vbias.reshape((self.dictsize, 1)))
				normC = visible_f1.sum(axis=0).reshape(( len(batchID[b]), ))
				visible_f1_pdf = visible_f1/normC
				visible_f1_sample = numpy.zeros(( self.dictsize, len(batchID[b]) ))
				for doc in xrange(len(batchID[b])):
					visible_f1_sample[:, doc] = numpy.random.multinomial(scaling[doc], visible_f1_pdf[:, doc], size=1)
				hidden_f2 = sigmoid( numpy.dot(visible_f1_sample.T, self.W)*self.bottomUp + numpy.outer(scaling, self.hbias)) # fantasy 2, probability
				hidden_f2 *= r # remove dropped units

				# no we're ready to perform CD updates:
				self.deltaW = self.deltaW * self.momentum + numpy.dot(visible.T, hidden_f1_sample) - numpy.dot(visible_f1_sample, hidden_f2)
				self.deltavbias = self.deltavbias * self.momentum + visible.sum(axis=0) - visible_f1_sample.sum(axis=1)
				self.deltahbias = self.deltahbias * self.momentum + hidden_f1_sample.sum(axis=0) - hidden_f2.sum(axis=0)

				# clip gradients!!
				if gradClip>0:
					self.deltaW[ self.deltaW > gradClip] = gradClip
					self.deltaW[ self.deltaW < (-1* gradClip)] = (-1* gradClip)

					self.deltavbias[ self.deltavbias > gradClip ] = gradClip
					self.deltavbias[ self.deltavbias < (-1*gradClip) ] = (-1 * gradClip)

					self.deltahbias[ self.deltahbias > gradClip ] = gradClip
					self.deltahbias[ self.deltahbias < (-1*gradClip) ] = (-1 * gradClip)

				# update parameters:
				self.W += self.deltaW * self.lr 
				self.vbias += self.deltavbias * self.lr
				self.hbias += self.deltahbias * self.lr

			if verbose:
				# print some results to see what type of progress we are making:
				print "\nMean squared error: %f \t (this is by all means probably not appropriate here)" %( numpy.sqrt( ((visible-visible_f1_pdf.T)**2).sum() )/batch_size )
				print "Cross-entropy: %f \t (probably more appropriate)\n" % getCrossEntropy(visible, visible_f1_pdf.T)

				if (e%10)==0:
					print "Current one step reconstruction for work 'fear'"
					print self.oneStepRecon("fear", adjustForDropOut=True)
					print "\n"

					print "Current one step reconstruction for work 'disorder'"
					print self.oneStepRecon("disorder", adjustForDropOut=True)
					print "\n\n"

				reconDist = []
				for x in self.wordVec:
					reconDist.append( list(self.oneStepRecon(x, n=len(self.wordVec), adjustForDropOut=True)).index(x))
				print "current 1-step recon dist: " +str(sum(reconDist)) + "\n"

		# finally correct weights to adjust for dropout:
		self.W *= self.dropOutProb
		self.vbias *= self.dropOutProb
		self.hbias *= self.dropOutProb
		self.weightsAdjusted = True

	def FreeEnergy(self, v, partition=False):
		"""
		calculate the free energy for unseen visible units v
		This can be calculated analytically when hidden units are binary!
		In order to compute the probability of visible unit we also require an estimate of the partition
		function - this is achieved using AIS

		INPUT:
			 - v: visible data (mean word counts across multi documents)
			 - partition: compute partition function (if necesary) and return log probability.
			   If false we return unnormalized probability

		"""

		if partition:
			if self.logZ==None:
				self.estimateZ_AIS(steps=1000, M=500)
			return -1*(numpy.log(1 + numpy.exp(numpy.dot(v.T, self.W) + v.sum()*self.hbias)).sum() + (numpy.dot(v.T, self.vbias))) - self.logZ
		else:
			return -1*(numpy.log(1 + numpy.exp(numpy.dot(v.T, self.W) + v.sum()*self.hbias)).sum() + (numpy.dot(v.T, self.vbias)))

	def estimateZ_AIS(self, steps, M):
		"""
		Estimate log partition function using Annealed Importance Sampling (AIS)

		INPUT:
			 - steps: number of steps to take (i.e., the number of auxiliary distrbutions to consider)
			 - M: number of particles to consider

		NOTE: this requires the RBM to have been previously trained!

		this code is based on Salakhutdinov's RBM_AIS.m code

		result is stored in self.logZ

		"""

		scaling = self.data.sum(axis=1).mean().astype(int) # mean document length - so we know how many visible activations to sample
		lw = self.h_units * numpy.log(2) * numpy.ones(M)
		visible_f1_pdf = numpy.ones( self.dictsize)/ self.dictsize # uniform initial distribution
		# sample negative data:
		visible_samp = numpy.zeros(( self.dictsize, M ))
		for doc in xrange(M):
			visible_samp[:, doc] = numpy.random.multinomial(scaling, visible_f1_pdf, size=1)

		W_h = numpy.dot( visible_samp.T, self.W )*self.bottomUp + scaling*self.hbias # part of free energy for negative data
		bias_v = numpy.dot(visible_samp.T, self.vbias) # other part of free energy equation

		for s in xrange(1,steps):
			b_k = float(s)/steps
			expW_h = numpy.exp( b_k * W_h) # weight adjusted hidden variable distribution (without normalization)
			lw += b_k*bias_v + (numpy.log(1+expW_h)).sum(axis=1) # this is effectively adding p^{*}_k(v_k) (following notation from Salak & Murray 2008)

			# apply transition (so sample hidden and visible once):
			hidden_f1 = sigmoid( numpy.dot(visible_samp.T, self.W)*b_k*self.bottomUp + b_k*scaling* self.hbias) # fantasy 1, probability
			hidden_f1_sample = (hidden_f1 > numpy.random.rand( M,  self.h_units) )*1

			# sample visible units:
			visible_f1 = numpy.exp(b_k * (numpy.dot(self.W, hidden_f1_sample.T) + self.vbias.reshape((self.dictsize, 1)) ) )
			normC = visible_f1.sum(axis=0).reshape(( M, ))
			visible_f1_pdf = visible_f1/normC
			for doc in xrange(M):
				visible_samp[:, doc] = numpy.random.multinomial(scaling, visible_f1_pdf[:, doc], size=1)

			# update W_h and bias_v
			W_h = numpy.dot( visible_samp.T, self.W ) + scaling * self.hbias # part of free energy for negative data
			bias_v = numpy.dot(visible_samp.T, self.vbias)

			expW_h = numpy.exp( b_k *( W_h))
			lw -= (b_k*bias_v + (numpy.log(1+expW_h)).sum(axis=1) ) # this is effectively subtracting p^{*}_k(v_{k+1}) 

		# add final term:
		expW_h = numpy.exp( ( W_h))
		lw += (bias_v + (numpy.log(1+expW_h)).sum(axis=1))

		# now collect all terms and return estimate of log partition function:
		self.logZ = logsumexp(lw) - numpy.log( M) # this is the log of the mean estimate
		self.logZ += self.h_units * numpy.log(2)
