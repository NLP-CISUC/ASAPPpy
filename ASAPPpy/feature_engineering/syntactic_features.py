'''
Module used to extract syntactic features from a given corpus.
'''

import spacy
import pandas as pd

from .lexical_features import jaccard_coefficient

def compute_pos(pipe_tags):
	""" Function used to compute the POS-tags difference between two sentences in the corpus """

	tags = ['adj', 'adv', 'art', 'conj-c', 'conj-s', 'intj', 'n', 'n-adj', 'num', 'pron-det', 'pron-indp', 'pron-pers', 'prop', 'prp', 'punc', 'v-fin', 'v-ger', 'v-inf', 'v-pcp']

	prepocessed_tagged_corpus = [line.split(' ') for line in pipe_tags]

	tagged_corpus_counts = []

	for i in range(0, len(prepocessed_tagged_corpus), 2):
		tagged_pair_counts = []

		# all_tags_difference = abs(len(prepocessed_tagged_corpus[i])-len(prepocessed_tagged_corpus[i+1]))
		# tagged_pair_counts.append(all_tags_difference)

		for tag in tags:
			temp_q = prepocessed_tagged_corpus[i].count(tag)
			temp_r = prepocessed_tagged_corpus[i+1].count(tag)

			difference = abs(temp_q-temp_r)
			tagged_pair_counts.append(difference)

		tagged_corpus_counts.append(tagged_pair_counts)

	# tags.insert(0, "all_tags")

	# convert list to Dataframe in order to use it with feature extraction
	tagged_corpus_counts = pd.DataFrame(tagged_corpus_counts)
	tagged_corpus_counts.columns = tags

	return tagged_corpus_counts

def dependency_parsing(parse_corpus):
	""" Function used to compute the syntactic dependencies between two sentences in the corpus """

	spacy_nlp = spacy.load('pt')

	dependency_values = []

	for i in range(0, len(parse_corpus), 2):
		sentence1_triplets = []
		sentence2_triplets = []

		parsed_sentence_1 = spacy_nlp(parse_corpus[i])
		parsed_sentence_2 = spacy_nlp(parse_corpus[i+1])

		for token in parsed_sentence_1:
			if token.dep_ != "ROOT" and token.dep_ != "punct":
				sentence1_triplets.append((token.text, token.head.text, token.dep_))

		for token in parsed_sentence_2:
			if token.dep_ != "ROOT" and token.dep_ != "punct":
				sentence2_triplets.append((token.text, token.head.text, token.dep_))

		pair_jc = jaccard_coefficient(set(sentence1_triplets), set(sentence2_triplets))

		dependency_values.append(pair_jc)

	# convert list to Dataframe in order to use it with feature extraction
	dependency_parsing_jc = pd.DataFrame(dependency_values)
	dependency_parsing_jc.columns = ["dependency_parsing_jc"]

	return dependency_parsing_jc
	