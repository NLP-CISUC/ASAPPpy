'''
Module used for feature extraction of a corpus.
'''
import os
import numpy as np
from xml.etree import cElementTree as ET
from pysts.assin.assineval.commons import read_xml_no_attributes

import pysts.tools as tl
from pysts import ROOT_PATH

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def bert_model(tokenizer, model, sentence_1, sentence_2):

	inputs = tokenizer.encode_plus(sentence_1, sentence_2, add_special_tokens=True, return_tensors="pt")

	answer_start_scores, answer_end_scores = model(**inputs)

	# convert both Torch tensors to numpy arrays
	start_np = answer_start_scores.detach().numpy()
	end_np = answer_end_scores.detach().numpy()

	# compute the multiplication between the two matrices
	score = np.matmul(np.expand_dims(start_np, -1), np.expand_dims(end_np, 1))

	score_flat = score.flatten()

	idx_sort = [np.argmax(score_flat)]

	start, end = np.unravel_index(idx_sort, score.shape)[1:]

	return score[0, start, end][0]

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

model = AutoModelForQuestionAnswering.from_pretrained('bert-base-multilingual-cased')

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
	result = bert_model(tokenizer, model, test_corpus[i], test_corpus[i+1])
	predicted_similarity.append(result)

# write output
tree = ET.parse(load_path)
root = tree.getroot()
for i in range(len(test_pairs)):
	pairs = root[i]
	pairs.set('entailment', "None")
	pairs.set('similarity', str(predicted_similarity[i]))

tree.write("test.xml", 'utf-8')

