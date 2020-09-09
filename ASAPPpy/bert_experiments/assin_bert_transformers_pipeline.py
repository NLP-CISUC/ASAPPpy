'''
Module used for feature extraction of a corpus.
'''
import os
from xml.etree import cElementTree as ET
from ASAPPpy.assin.assineval.commons import read_xml_no_attributes

import ASAPPpy.tools as tl
from ASAPPpy import ROOT_PATH

from transformers import pipeline

nlp = pipeline(model='bert-base-multilingual-cased', task="question-answering")

# extract labels
test_pairs = []

load_path = os.path.join(ROOT_PATH, 'datasets', 'assin', 'assin2', 'assin2-blind-test.xml')

test_pairs.extend(read_xml_no_attributes(load_path))

# extract training features
test_corpus = tl.read_corpus(test_pairs)

number_of_pairs = int(len(test_corpus)/2)

predicted_similarity = []

for i in range(0, len(test_corpus), 2):
	if i == 0:
		print('Variant %d/%d' % (1, number_of_pairs), end='\r')
	else:
		print('Variant %d/%d' % (int((i+1)/2), number_of_pairs), end='\r')
	result = nlp(question=test_corpus[i], context=test_corpus[i+1])
	predicted_similarity.append(result['score'])

# write output
tree = ET.parse(load_path)
root = tree.getroot()
for i in range(len(test_pairs)):
	pairs = root[i]
	pairs.set('entailment', "None")
	pairs.set('similarity', str(predicted_similarity[i]))

tree.write("test.xml", 'utf-8')

