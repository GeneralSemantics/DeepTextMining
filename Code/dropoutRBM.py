### implementation of RESTRICTED BOLTZMANN MACHINE
#
# this is used further down the stack of our DBM
# note that the following code performed CD-1 to estimate gradients
#
#

import numpy, scipy, os
import cPickle as pickle


def sigmoid(x):
    """
    sigmoid activation function
    """
    return (1 + scipy.tanh(x/2))/2


class dropoutRBM(object):
	"""

	class for restricted boltzmann machine 

	"""

	def __init__(self, data, validation_data):
		"""
		initialize RBM object
		
		input:
			 - data: incidence matrix - each row a document, each column the number of times a word appears in that document 
			 - validation_data: same format as data, used to track validation performance

		"""

		self.data = data
		self.val_data = validation_data
		self.v_units = data.shape[1] # number of columns indicates number of words


	def generateHiddenSamples(self,  n=1e5):
		"""
		Generate samples from hidden units. This is needed when training of DBMs as these samples are used 
		to train further RBMs in the stack

		n is the number of samples to generate (should be a big number!)

		"""

		hiddenSamples = numpy.zeros((int(n), self.h_units))# each row will be a sample from hidden units
		ii = numpy.random.randint(low=0, high=self.data.shape[0]-1, size=n )
		visible = self.data[ii,:]
		hprob = sigmoid( numpy.dot(visible, self.W)*self.bottomUp +  self.hbias)
		hiddenSamples = (hprob > numpy.random.rand( n,  self.h_units) )*1 

		return hiddenSamples

	def train(self, h_units, epochs, batch_size, lr, momentum, dropOutProb = .5, GibbsSteps = 5, numpyRNG=None, layer="top"):
		"""
		train a DBM using CD with dropout

		input:
			 - h_units: number of hidden units to employ
			 - epochs: number of training epochs 
			 - batch_size: number of documents in each batch update
			 - lr: learning rate 
			 - momentum: momentum rate (set to 0 for standard GD)
			 - dropOutProb: probability of dropping hidden unit
			 - layer: layer=top indicates this is the top layer of a DBM, weights will be adjusted accordingly (likewise for layer=bottom)


		"""
		self.h_units = h_units
		self.lr = lr
		self.momentum = momentum
		self.dropOutProb = dropOutProb
		self.weightsAdjusted = False # indicates if weights have been adjusted (to correct for dropout)
		if layer=="top":
			print "training for top layer in DBM... weights will be adjusted accordingly"
			self.topDown = 2
			self.bottomUp = 1
		elif layer=="bottom":
			print "training for bottom layer in DBM... weights will be adjusted accordingly"
			self.topDown = 1
			self.bottomUp = 2
		else:
			print "training vanilla RBM"
			self.topDown = 1
			self.bottomUp = 1

		if numpyRNG is None:
			self.numpy_rng = numpy.random.RandomState(1234)
		else:
			self.numpy_rng = numpyRNG

		self.W = self.numpy_rng.uniform( low=-4 * numpy.sqrt(6. / (self.h_units + self.v_units)),
					  high=4 * numpy.sqrt(6. / (self.h_units + self.v_units)),
					  size=(self.v_units, self.h_units))

		self.hbias = numpy.zeros((self.h_units, ))
		self.vbias = numpy.zeros((self.v_units,))

		self.deltaW = numpy.zeros(( self.v_units, self.h_units ))
		self.deltahbias = numpy.zeros((self.h_units, ))
		self.deltavbias = numpy.zeros((self.v_units, ))

		self.Errors = [] # store errors to study/plot later

		# prepare data:
		batch_num = int(numpy.floor(self.data.shape[0] / batch_size))

		# split documents into batches:
		ii = range(self.data.shape[0])
		batchID = [ii[i::batch_num] for i in range(batch_num)]

		print "training replicated RBM using CD-1..."
		# now we iterative of epochs and over batches:
		for e in xrange(epochs):
			print "running epoch %s" %(e+1)
			for b in xrange(batch_num):
				visible = numpy.array(self.data[batchID[b]]) # select visible units

				# sample dropped units (redo for each batch!):
				r = ( self.dropOutProb > numpy.random.rand( self.h_units ))*1

				## positive phase:
				hidden_f1 = sigmoid( numpy.dot(visible, self.W)*self.bottomUp + self.hbias) # fantasy 1, probability
				hidden_f1_sample = (hidden_f1 > numpy.random.rand( len(batchID[b]),  self.h_units) )*1 # fantasy 1, sample - needed for CD
				hidden_f1_sample *= r # remove dropped units

				## negative phase (CD-k):
				hidden_f1_sample_neg = numpy.copy(hidden_f1_sample)
				for k in xrange(GibbsSteps):
					visible_f1_prob = sigmoid(numpy.dot(self.W, hidden_f1_sample_neg.T)*self.topDown + self.vbias.reshape((self.v_units, 1))).T
					# not sure if I should sample from visible_f1_prob here...
					hidden_cd_prob = sigmoid(numpy.dot(visible_f1_prob, self.W)*self.bottomUp + self.hbias) * r
					hidden_f1_sample_neg = (hidden_cd_prob > numpy.random.rand(len(batchID[b]), self.h_units) ) * 1
					hidden_f1_sample_neg *= r

				# no we're ready to perform CD updates:
				self.deltaW = self.deltaW * self.momentum + numpy.dot(visible.T, hidden_f1_sample) - numpy.dot(visible_f1_prob.T, hidden_cd_prob)
				self.deltavbias = self.deltavbias * self.momentum + visible.sum(axis=0) - visible_f1_prob.sum(axis=0)
				self.deltahbias = self.deltahbias * self.momentum + hidden_f1_sample.sum(axis=0) - hidden_cd_prob.sum(axis=0)

				# update parameters:
				self.W += self.deltaW * self.lr 
				self.vbias += self.deltavbias * self.lr
				self.hbias += self.deltahbias * self.lr

			# print some results to see what type of progress we are making:
			print "\nMean squared error: %f \t (appropriate for Gaussian/binary visible units)" %( numpy.sqrt( ((visible-visible_f1_prob)**2).sum() )/batch_size )
			self.Errors.append( ( numpy.sqrt( ((visible-visible_f1_prob)**2).sum() )/batch_size ) )

		# correct weights to account for dropout:
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
			return -1*(numpy.log(1 + numpy.exp(numpy.dot(v.T, self.W) + self.hbias)).sum() + (numpy.dot(v.T, self.vbias))) - self.logZ
		else:
			return -1*(numpy.log(1 + numpy.exp(numpy.dot(v.T, self.W) + self.hbias)).sum() + (numpy.dot(v.T, self.vbias)))

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

		lw = self.h_units * numpy.log(2) * numpy.ones(M)
		# we start with a sample from a uniform distribution (RBM where all parameters are 0)
		visible_samp = (0.5 > numpy.random.rand( M,  self.v_units) )*1

		W_h = numpy.dot( visible_samp, self.W )*self.bottomUp + self.hbias # part of free energy for negative data
		bias_v = numpy.dot(visible_samp, self.vbias) # other part of free energy equation

		for s in xrange(1,steps):
			b_k = float(s)/steps
			expW_h = numpy.exp( b_k * W_h) # weight adjusted hidden variable distribution (without normalization)
			lw += b_k*bias_v + (numpy.log(1+expW_h)).sum(axis=1) # this is effectively adding p^{*}_k(v_k) (following notation from Salak & Murray 2008)

			# apply transition (so sample hidden and visible once):
			hidden_f1 = sigmoid( numpy.dot(visible_samp, self.W)*b_k*self.bottomUp + b_k* self.hbias) # fantasy 1, probability
			hidden_f1_sample = (hidden_f1 > numpy.random.rand( M,  self.h_units) )*1

			# sample visible units:
			visible_f1_pdf = sigmoid(numpy.dot(self.W, hidden_f1_sample.T)*self.topDown + self.vbias.reshape((self.v_units, 1))).T
			visible_samp = ( visible_f1_pdf > numpy.random.rand( M, self.v_units))*1

			# update W_h and bias_v
			W_h = numpy.dot( visible_samp, self.W ) + self.hbias # part of free energy for negative data
			bias_v = numpy.dot(visible_samp, self.vbias)

			expW_h = numpy.exp( b_k *( W_h))
			lw -= (b_k*bias_v + (numpy.log(1+expW_h)).sum(axis=1) ) # this is effectively subtracting p^{*}_k(v_{k+1}) 

		# add final term:
		expW_h = numpy.exp( ( W_h))
		lw += (bias_v + (numpy.log(1+expW_h)).sum(axis=1))

		# now collect all terms and return estimate of log partition function:
		self.logZ = scipy.misc.logsumexp(lw) - numpy.log( M) # this is the log of the mean estimate
		self.logZ += self.h_units * numpy.log(2)


