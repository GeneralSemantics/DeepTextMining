### Collect text from pubmed
#
# We do the following:
#	 - get DOIs from the neurosynth database
#	 - use these to search pubmet and get text using Biopython
# 	 - save abstract text in "Data/Raw" folder as a pickle file
#
#

import numpy, os
import pandas as pd
import re
from Bio import Entrez
import cPickle as pickle
import string 
import textmining # module to build incidence matrix!
from nltk.corpus import stopwords 

# helper functions to clean individual workds:
def cleanWord(w):
	"""
	clean word, w, by making lower case and removing punctution
	"""

	try:
		w=str(w).replace('-', ' ') # for hyphenated words
	except UnicodeEncodeError:
		w = ''
	return w.translate(string.maketrans("",""), string.punctuation).lower()

# define directories:
DataDir = "../Data" # define data directory - may require manual change
StoreDir = "../Data/Raw" # define location to save abstracts

# define some things for biopython:
Entrez.email = "" # enter email here

# load in data from neurosynth first:
os.chdir(DataDir)
data = pd.read_csv('database.txt', sep='\t', index_col=0) # this is neurosynth database

# load in vocabulary:
Vocab = set(pickle.load(open("Vocabulary_text8.p", "rb")))

# keep track of papers we've collected:
DOI = [x.split('_')[0] for x in os.listdir(StoreDir) if 'abstract' in x]

# start collecting:

for i in range(data.shape[0]):
	pmid = str(data.take([i])['doi']).split("\n")[1].split(" ")[0]

	if pmid in DOI:
		pass
	else:
		print "Collecting "+ str(len(DOI))+"th article"
		try:
			# collect abstract using pubmed API:
			handle = Entrez.efetch(db='pubmed', id=str(pmid), retmode='text', rettype='full')
			abstract = handle.read()

			# do some cleaning:
			newText = abstract.split('\n\n')[4].replace('\n', ' ')
			newText2 = [cleanWord(x) for x in newText.split(' ')]
			newText2 = [x for x in newText2 if x in Vocab] # remove words not in vocabulary
			pickle.dump( ' '.join(newText2), open(StoreDir + "/" + pmid+"_abstract.p", "wb"))
		except:
			print "Error with this article... DOI: " + str(pmid)



# now build incidence matrix:
os.chdir(StoreDir)
files = [x for x in files if "abstract" in x]
D = textmining.TermDocumentMatrix()
counter = 0
for x in files:
	doc = pickle.load(open(x, "rb"))
	D.add_doc(doc)

os.chdir(DataDir)
D.write_csv("abstractIncidence.csv")

# now we go through and clean incidence matrix:
with open("abstractIncidence.csv") as f:
		reader = csv.reader(f)
		Xword = numpy.array(next(reader))

# remove words with 2 or fewer letters (i.e., "an", "to", "ie", etc)
ii = [x for x in range(len(Xword)) if len(Xword[x]) > 2]  #[ True if len(x) > 2 else False for x in Xword[:5]  ]
Dat = Dat[:, ii]
Xword = Xword[ii]

# remove words that occur too often or note often enough:
ii = ((Dat.sum(axis=0) > 50) & (Dat.sum(axis=0) < 12500))
Dat = Dat[:, ii]
Xword2 = Xword[ii]

# and remove stopwords:
ii = [x for x in range(len(Xword2)) if Xword2[x] not in stopwords.words('english')]
Dat = Dat[:, ii]
Xword2 = Xword2[ii]

# load other stopwords:
os.chdir("/media/ricardo/TheBadBoy/Neurosynth/Text/Incidence Matrix")
StopWords = numpy.genfromtxt("stopwords.txt", dtype='str')
ii = [x for x in range(len(Xword2)) if Xword2[x] not in StopWords]
Dat = Dat[:, ii]
Xword2 = Xword2[ii]

# take the 2000 top words
ii = Dat.mean(axis=0).argsort()[::-1][:1000] # contains 82% of all word occurances!
Dat = Dat[:, ii]
Xword2 = Xword2[ii]

os.chdir(DataDir)
pickle.dump(Dat, open("WordIncidence_1000.p", "wb")) # this is the incidence matrix
pickle.dump(Xword2, open("Words_1000.p", "wb")) # this is the vector of words used
