'''
The Chatbot Module
'''

import os
from xml.etree import cElementTree as ET
from ASAPPpy.assin.assineval.commons import read_xml_no_attributes

import ASAPPpy.tools as tl
from ASAPPpy import ROOT_PATH

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, models
import scipy.spatial

def bert_model(model, sentence_1, sentence_2):

	sentence_embedding_1 = model.encode([sentence_1])
	sentence_embedding_2 = model.encode([sentence_2])
	# sentence_embedding_1 = sentence_embedding_1[0]
	# sentence_embedding_2 = sentence_embedding_2[0]

	similarity = cosine_similarity(sentence_embedding_1, sentence_embedding_2)

	return similarity[0][0]

def chatbot():
	model = SentenceTransformer('distiluse-base-multilingual-cased')

	# Use BERT for mapping tokens to embeddings
	# word_embedding_model = models.BERT('bert-large-portuguese-cased')

	# Apply mean pooling to get one fixed sized sentence vector
	# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
	# 							pooling_mode_mean_tokens=True,
	# 							pooling_mode_cls_token=False,
	# 							pooling_mode_max_tokens=False)

	# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

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
		result = bert_model(model, test_corpus[i], test_corpus[i+1])
		predicted_similarity.append(result)

	# write output
	tree = ET.parse(load_path)
	root = tree.getroot()
	for i in range(len(test_pairs)):
		pairs = root[i]
		pairs.set('entailment', "None")
		pairs.set('similarity', str(predicted_similarity[i]))

	tree.write("test.xml", 'utf-8')

if __name__ == '__main__':
	chatbot()
	# read_faqs_variants()
