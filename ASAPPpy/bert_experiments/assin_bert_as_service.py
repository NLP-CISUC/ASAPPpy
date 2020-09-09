'''
The Chatbot Module
'''

import os
from xml.etree import cElementTree as ET
from ASAPPpy.assin.assineval.commons import read_xml_no_attributes

import ASAPPpy.tools as tl
from ASAPPpy import ROOT_PATH

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import scipy.spatial

import numpy as np
from bert_serving.client import BertClient
# from termcolor import colored

# prefix_q = '##### **Q:** '
# topk = 5

# with open('README.md') as fp:
# 	questions = [v.replace(prefix_q, '').strip() for v in fp if v.strip() and v.startswith(prefix_q)]
# 	print(questions)
# 	print('%d questions loaded, avg. len of %d' % (len(questions), np.mean([len(d.split()) for d in questions])))

# with BertClient(port=4000, port_out=4001) as bc:
# 	doc_vecs = bc.encode(questions)

# 	while True:
# 		query = input(colored('your question: ', 'green'))
# 		query_vec = bc.encode([query])[0]
# 		# compute normalized dot product as score
# 		score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
# 		topk_idx = np.argsort(score)[::-1][:topk]
# 		print('top %d questions similar to "%s"' % (topk, colored(query, 'green')))
# 		for idx in topk_idx:
# 			print('> %s\t%s' % (colored('%.1f' % score[idx], 'cyan'), colored(questions[idx], 'yellow')))

def bert_model(model, sentence_1, sentence_2):

	sentence_embedding_1 = model.encode([sentence_1])[0]
	sentence_embedding_2 = model.encode([sentence_2])[0]
	# sentence_embedding_1 = sentence_embedding_1[0]
	# sentence_embedding_2 = sentence_embedding_2[0]
	similarity = np.sum(sentence_embedding_1 * sentence_embedding_2) / np.linalg.norm(sentence_embedding_2)

	return similarity

def chatbot():
	model = BertClient(port=4000)

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
