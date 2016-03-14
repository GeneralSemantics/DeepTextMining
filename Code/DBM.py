## DEEP BOLTZMANN MACHINE
#
#
# this class implements a 3 layer DBM with softmax visible units
#
#

import scipy
import numpy
import pylab as plt


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

class softmaxDBM_3layer(object):
	"""
	class of deep boltzmann machines with softmax visible layers and 2 hidden units (so 3 layers in total)

	initialization assumes that pretraining has already taken place!

	"""

	def __init__(self, softmaxLayer, RBMlayer):
		"""
		intialize softmax DBM object
		"""	
		self.softmaxLayer = softmaxLayer
		self.l2layer = RBMlayer

	def train(self, epochs, M, batch_size, lr, momentum, GibbsSteps=5, numpyRNG=None, decRate=1, verbose=False):
		"""

		fine tuning of DBM using variation inference and stochastic approximation

		INPUT:
			- epochs: number of training epochs
			- M: number of sample particles (using for SA step)
			- batch_size
			- lr, momentum: learning rate and momentum. momentum is increased after 5 epochs
			- GibbsSteps: number of Gibbs steps to take in SA approx to data indep stats (set to 5 following Salakhutdinov)
			- decRate: rate at which we decrease (set to 1 for a constant learning rate - should be less than 1)

		"""

		# set random number generator:
		if numpyRNG is None:
			self.numpy_rng = numpy.random.RandomState(1234)
		else:
			self.numpy_rng = numpyRNG

		self.lr = lr
		self.momentum = momentum
		self.M = M
		self.crossEnt = []

		# store weight matrices (very inefficient as im effetively storing them twice...)
		self.W_l1 = numpy.copy(self.softmaxLayer.W)
		self.W_l2 = numpy.copy(self.l2layer.W)

		if False:
			self.hbias_l1 = numpy.zeros((self.softmaxLayer.h_units, ))#numpy.copy(self.softmaxLayer.hbias)
			self.vbias_l1 = numpy.zeros((self.softmaxLayer.dictsize, ))#numpy.copy(self.softmaxLayer.vbias)	
			self.hbias_l2 = numpy.zeros((self.l2layer.h_units, )) #numpy.copy(self.l2layer.hbias)
			self.vbias_l2 = numpy.zeros((self.l2layer.v_units, )) #numpy.copy(self.l2layer.vbias)
		else:
			# initialize to pretrained biases:
			self.hbias_l1 = numpy.copy(self.softmaxLayer.hbias)
			self.vbias_l1 = numpy.copy(self.softmaxLayer.vbias)	
			self.hbias_l2 = numpy.copy(self.l2layer.hbias)
			self.vbias_l2 = numpy.copy(self.softmaxLayer.hbias) #numpy.copy(self.l2layer.vbias)


		# save current bests:
		self.CB_W_l1 = numpy.copy(self.W_l1)
		self.CB_W_l2 = numpy.copy(self.W_l2)
		self.CB_vbias_l1 = numpy.copy(self.vbias_l1)
		self.CB_vbias_l2 = numpy.copy(self.vbias_l2)
		self.CB_hbias_l1 = numpy.copy(self.hbias_l1)
		self.CB_hbias_l2 = numpy.copy(self.hbias_l2)
		self.currentBest = 1000*1000

		# build sample particles:
		self.visibleSample = self.softmaxLayer.data[ numpy.random.choice(self.softmaxLayer.data.shape[0], self.M), :] # randomly sample visible units for this (I assume/hope this is OK)
		# use pretrained weights to set initial estimates
		scaling_particle = self.visibleSample.sum(axis=1)
		h1_activation = sigmoid( (numpy.dot(self.visibleSample, self.W_l1)*2 + numpy.outer(scaling_particle, self.hbias_l1 ) ))
		self.hiddenl1Sample = (h1_activation > numpy.random.rand( self.M,  self.softmaxLayer.h_units) )*1 #
		
		# finally sample h_2 | h_1
		h2_activation = sigmoid( numpy.dot(self.hiddenl1Sample, self.W_l2) + self.hbias_l2 )
		self.hiddenl2Sample = (h2_activation > numpy.random.rand( self.M,  self.l2layer.h_units) )*1 #


		#self.hiddenl1Sample = numpy.random.binomial(1, .5, size=self.M*self.softmaxLayer.h_units).reshape((self.M, self.softmaxLayer.h_units)) # each row a sample particle!
		#self.hiddenl2Sample = numpy.random.binomial(1, .5, size=self.M*self.l2layer.h_units).reshape((self.M, self.l2layer.h_units))

		# define increments:
		self.deltaW_l1 = numpy.zeros(( self.softmaxLayer.dictsize, self.softmaxLayer.h_units ))
		self.deltahbias_l1 = numpy.zeros((self.softmaxLayer.h_units, ))
		self.deltavbias_l1 = numpy.zeros((self.softmaxLayer.dictsize, ))

		self.deltaW_l2 = numpy.zeros(( self.l2layer.v_units, self.l2layer.h_units ))
		self.deltahbias_l2 = numpy.zeros((self.l2layer.h_units, ))
		self.deltavbias_l2 = numpy.zeros((self.l2layer.v_units, ))

		# finally we prepare the data:
		batch_num = int(numpy.floor(self.softmaxLayer.data.shape[0] / batch_size))

		# split documents into batches:
		ii = range(self.softmaxLayer.data.shape[0])
		batchID = [ii[i::batch_num] for i in range(batch_num)]

		print "training replicated softmax DBM..."
		# now we iterative of epochs and over batches:
		for e in xrange(epochs):
			print "running epoch %s" %(e+1)
			crossEnt = []
			
			#if e > 5:
			#	self.momentum = .9 # following Salakdinov
			
			self.lr *= decRate
			for b in xrange(batch_num):
				visible = self.softmaxLayer.data[batchID[b]] # select visible units
				scaling = visible.sum(axis=1) # scaling required for hidden variables
				n = visible.shape[0]

				# RUN MEAN FIELD TO GET DATA DEPENDENT STATISTICS
				mu1, mu2 = self.runMeanField(visible, scaling) # mu1, mu2 will be used to get data-dep statistics!
				mu1_binary = (mu1 >  numpy.random.rand( n, self.softmaxLayer.h_units ) )*1
				# store CD-1 pdf to report changes in cross-entropy
				visible_f1 = numpy.exp(numpy.dot(self.W_l1, mu1.T) + self.vbias_l1.reshape((self.softmaxLayer.dictsize,1)))
				normC = visible_f1.sum(axis=0).reshape(( len(batchID[b]), ))
				visible_f1_pdf = visible_f1/normC # this will be used to report cross-entropy later!
				crossEnt.append(getCrossEntropy(visible, visible_f1_pdf.T))

				# RUN SA TO GET DATA INDEPENDENT STATISTICS 
				for rep in xrange(GibbsSteps):
					# first sample v | h:
					visible_samp = numpy.exp( numpy.dot(self.W_l1, self.hiddenl1Sample.T ).T + self.vbias_l1)
					normC = visible_samp.sum(axis=1).reshape(( self.M, ))
					visible_samp_pdf = (visible_samp.T/normC).T
					scaling_particle = self.visibleSample.sum(axis=1)
					# now we sample:
					for doc in xrange(self.M):
						self.visibleSample[ doc, :] = numpy.random.multinomial(scaling_particle[doc], visible_samp_pdf[doc,:], size=1)

					# now sample h_1 | v, h_2
					h1_activation = sigmoid( (numpy.dot(self.visibleSample, self.W_l1) + numpy.outer(scaling_particle, self.hbias_l1)) + (numpy.dot(self.hiddenl2Sample, self.W_l2.T) + self.vbias_l2 ) )
					self.hiddenl1Sample = (h1_activation > numpy.random.rand( self.M,  self.softmaxLayer.h_units) )*1 #
					
					# finally sample h_2 | h_1
					h2_activation = sigmoid( numpy.dot(self.hiddenl1Sample, self.W_l2) + self.hbias_l2 )
					self.hiddenl2Sample = (h2_activation > numpy.random.rand( self.M,  self.l2layer.h_units) )*1 #

				# UPDATE WEIGHTS
				self.deltaW_l1 = self.deltaW_l1*self.momentum + ((numpy.dot(visible.T, mu1_binary)/(float(n))) - (numpy.dot(self.visibleSample.T, self.hiddenl1Sample)/(float(self.M)))) * float(n)
				self.deltahbias_l1 = self.deltahbias_l1*self.momentum + ( mu1_binary.mean(axis=0) - self.hiddenl1Sample.mean(axis=0) ) * float(n)
				self.deltavbias_l1 = self.deltavbias_l1*self.momentum + ( visible.mean(axis=0) - self.visibleSample.mean(axis=0)) * float(n)

				self.deltaW_l2 = self.deltaW_l2*self.momentum + ((numpy.dot(mu1.T, mu2)/(float(n)) - numpy.dot(self.hiddenl1Sample.T, self.hiddenl2Sample)/float(self.M) )) * float(n)
				self.deltahbias_l2 = self.deltahbias_l2*self.momentum + ( mu2.mean(axis=0) - self.hiddenl2Sample.mean(axis=0) ) * float(n)
				self.deltavbias_l2 = self.deltavbias_l2*self.momentum + ( mu1.mean(axis=0) - self.hiddenl1Sample.mean(axis=0)) * float(n)

				self.W_l1 += self.lr * self.deltaW_l1
				self.W_l2 += self.lr * self.deltaW_l2
				self.hbias_l1 += self.lr * self.deltahbias_l1
				self.hbias_l2 += self.lr * self.deltahbias_l2
				self.vbias_l1 += self.lr * self.deltavbias_l1
				self.vbias_l2 += self.lr * self.deltavbias_l2

			# print some reconstruction error statistics:
			if verbose:
				print "Cross-entropy:\t"+ str(numpy.mean(crossEnt)) + " ("+  str(numpy.round(numpy.std(crossEnt), 2)) + ")"
				self.crossEnt.append(numpy.mean(crossEnt))

				if (e%10)==0:
					print "Current one step reconstruction for work 'fear'"
					print self.oneStepRecon("fear", n=15, mode="dbm")
					print "\n"

					print "Current one step reconstruction for work 'disorder'"
					print self.oneStepRecon("disorder", n=15, mode="dbm")

					print "Current one step reconstruction for work 'pain'"
					print self.oneStepRecon("pain", n=15, mode="dbm")

					print "indexes for some words"
					print "fear \t %d" % list(self.oneStepRecon("fear", n=1000, mode="dbm")).index("fear")
					print "amygdala \t %d" % list(self.oneStepRecon("amygdala", n=1000, mode="dbm")).index("amygdala")
					print "pain \t %d" % list(self.oneStepRecon("pain", n=1000, mode="dbm")).index("pain")
					print "autism \t %d" % list(self.oneStepRecon("autism", n=1000, mode="dbm")).index("autism")
					print "disorder \t %d" % list(self.oneStepRecon("disorder", n=1000, mode="dbm")).index("disorder")
					print "memory \t %d" % list(self.oneStepRecon("memory", n=1000, mode="dbm")).index("memory")
					print "motor \t %d" % list(self.oneStepRecon("motor", n=1000, mode="dbm")).index("motor")
					print "\n"

			# calculate reconstruction distance:
			reconDist = []
			for x in self.softmaxLayer.wordVec:
				reconDist.append( list(self.oneStepRecon(x, n=1000, mode="dbm", weightsAdjusted=True)).index(x))
			reconDist = numpy.array(reconDist)

			if sum(reconDist) <= self.currentBest:
				self.CB_W_l1 = numpy.copy(self.W_l1)
				self.CB_W_l2 = numpy.copy(self.W_l2)
				self.CB_vbias_l1 = numpy.copy(self.vbias_l1)
				self.CB_vbias_l2 = numpy.copy(self.vbias_l2)
				self.CB_hbias_l1 = numpy.copy(self.hbias_l1)
				self.CB_hbias_l2 = numpy.copy(self.hbias_l2)
				self.currentBest = sum(reconDist)
				if verbose:
					print "%d words in top 10 reconstructions" % sum(reconDist < 10)
					print "%d words in top 50 reconstructions" % sum(reconDist < 50)
					print "current reconDist sum: %d" % sum(reconDist)


	def runMeanField(self, visible, scaling, tol=1e-3, miter=100, intialize="random", verbose=False):
		"""
		run mean field approximation for data-dependent statistics 

		INPUT:
			- visible: set of visible word count (multinomial units)
			- scaling: scaling for visible documents
			- tol, miter: convergence tolerance and max number of iterations
			- intialize: how to initialize hidden variational layers: either random or zero
		visible is set of visible word counts and scaling is the corresponding scaling

		"""
		# initialize with a single upward pass:
		if intialize=="random":
			mu2 = numpy.random.rand( self.l2layer.h_units) 
		else:
			mu2 = numpy.zeros((self.l2layer.h_units, ))

		mu1 = sigmoid( numpy.dot(visible, self.W_l1) +  numpy.outer(scaling, self.hbias_l1) + (numpy.dot(mu2, self.W_l2.T) + self.vbias_l2 ) )	
		mu2 = sigmoid( numpy.dot(mu1, self.W_l2) + self.hbias_l2 )

		mu1_old = mu1; mu2_old = mu2 # for convergence checking
		iter_ = 0
		conv = False # convergence flag
		while (conv==False):
			# perform one cycle of variational mean-field inference
			mu1 = sigmoid(  (numpy.dot(visible, self.W_l1) + numpy.outer(scaling, self.hbias_l1)) + (numpy.dot(mu2, self.W_l2.T) + self.vbias_l2 ) )
			mu2 = sigmoid( numpy.dot(mu1, self.W_l2) + self.hbias_l2 )

			if ((((abs(mu2-mu2_old)).sum() < tol) & ((abs(mu1-mu1_old)).sum()< tol)) | (miter < iter_)):
				conv = True
			else:
				iter_ += 1
				mu1_old = numpy.copy(mu1)
				mu2_old = numpy.copy(mu2) 

		if verbose:	print "converged in " +str(iter_) + " iterations"
		return mu1, mu2


	def oneStepRecon(self, word, n=10, mode="original", weightsAdjusted=True ):
		"""
		one step reconstruction of word

		if original==True, we use the weights from pretraining
		weightsAdjusted indicates if we should adjust the weights as in pretraining

		"""

		if mode=="original":
			ii = list(self.softmaxLayer.wordVec).index(word) # word index
			softmaxAct = sigmoid(self.softmaxLayer.W[ii,:]*self.softmaxLayer.bottomUp +self.softmaxLayer.hbias) # middle layer activation)
			topAct = sigmoid( numpy.dot(softmaxAct, self.l2layer.W) + self.l2layer.hbias) 
			softmaxAct_down = sigmoid( numpy.dot(self.l2layer.W, topAct)*self.l2layer.topDown + self.l2layer.vbias)
			vprob = (  numpy.dot(self.softmaxLayer.W, softmaxAct_down) +self.softmaxLayer.vbias)#+ my_rsm.vbias)
		elif mode=="best":
				ii = list(self.softmaxLayer.wordVec).index(word) # word index
				softmaxAct = sigmoid(self.CB_W_l1[ii,:] +self.CB_hbias_l1) # middle layer activation)
				topAct = sigmoid( numpy.dot(softmaxAct, self.CB_W_l2) + self.CB_hbias_l2) 
				softmaxAct_down = sigmoid( numpy.dot(self.CB_W_l2, topAct) + self.CB_vbias_l2)
				vprob = (  numpy.dot(self.CB_W_l1, softmaxAct_down) +self.CB_vbias_l1)#+ my_rsm.vbias)
		else:
			if weightsAdjusted==True:
				ii = list(self.softmaxLayer.wordVec).index(word) # word index
				softmaxAct = sigmoid(self.W_l1[ii,:]*self.softmaxLayer.bottomUp +self.hbias_l1) # middle layer activation)
				topAct = sigmoid( numpy.dot(softmaxAct, self.W_l2) + self.hbias_l2) 
				softmaxAct_down = sigmoid( numpy.dot(self.W_l2, topAct)*self.l2layer.topDown + self.vbias_l2)
				vprob = (  numpy.dot(self.W_l1, softmaxAct_down) +self.vbias_l1)#+ my_rsm.vbias)
			else:
				ii = list(self.softmaxLayer.wordVec).index(word) # word index
				softmaxAct = sigmoid(self.W_l1[ii,:] +self.hbias_l1) # middle layer activation)
				topAct = sigmoid( numpy.dot(softmaxAct, self.W_l2) + self.hbias_l2) 
				softmaxAct_down = sigmoid( numpy.dot(self.W_l2, topAct) + self.vbias_l2)
				vprob = (  numpy.dot(self.W_l1, softmaxAct_down) +self.vbias_l1)#+ my_rsm.vbias)

		return self.softmaxLayer.wordVec[vprob.argsort()[::-1][:n]]
