'''
Module used to extract semantic features from a given corpus.
'''

import os
import collections

import pandas as pd

from ASAPPpy.scripts.tools import preprocessing

def compute_ner(ners_corpus):
	""" Function used to compute the NERs difference between two sentences in the corpus """

	usable_ners = ['B-ABSTRACCAO', 'B-ACONTECIMENTO', 'B-COISA', 'B-LOCAL', 'B-OBRA', 'B-ORGANIZACAO', 'B-OUTRO', 'B-PESSOA', 'B-TEMPO', 'B-VALOR']

	ners_corpus_counts = []

	for i in range(0, len(ners_corpus), 2):
		ners_pair_counts = []
		total_ners_q = 0
		total_ners_r = 0

		for ner in usable_ners:
			temp_q = ners_corpus[i].count(ner)
			temp_r = ners_corpus[i+1].count(ner)

			difference = abs(temp_q-temp_r)
			ners_pair_counts.append(difference)

			total_ners_q += temp_q
			total_ners_r += temp_r

		all_ners_difference = abs(total_ners_q-total_ners_r)
		ners_pair_counts.insert(0, all_ners_difference)

		ners_corpus_counts.append(ners_pair_counts)

	usable_ners.insert(0, "all_ners")

	# convert list to Dataframe in order to use it with feature extraction
	ners_corpus_counts = pd.DataFrame(ners_corpus_counts)
	ners_corpus_counts.columns = usable_ners

	return ners_corpus_counts

def semantic_relations_coefficient(sentence1, sentence2, dictionaries_list):
	""" Auxiliar function used to compute the semantic relations coefficient between two sentences in the corpus """

	counters = len(dictionaries_list) * [0]
	normalization_counters = len(dictionaries_list) * [0]

	for count, relation in enumerate(dictionaries_list):
		for word in sentence1:
			if word in relation.keys():
				for definition in relation[word]:
					if definition in sentence2:
						counters[count] += 1
			if counters[count] != 0:
				normalization_counters[count] += 1

	normalized_counters = []

	for i in range(len(counters)):
		if counters[i] == 0 or normalization_counters[i] == 0:
			normalized_counters.append(0)
		else:
			normalized_counters.append(counters[i]/normalization_counters[i])

	# used to create a single feature for all semantic relations
	# if sum(normalized_counters) == 0:
	# 	relations_coefficient = 0
	# else:
	# 	relations_coefficient = sum(normalized_counters)/len(dictionaries_list)

	# return relations_coefficient

	# used to create a feature for each semantic relation
	if sum(normalized_counters[0:3]) == 0:
		antonym_feature = 0
	else:
		antonym_feature = sum(normalized_counters[0:3])/4

	if sum(normalized_counters[4:7]) == 0:
		synonym_feature = 0
	else:
		synonym_feature = sum(normalized_counters[4:7])/4

	if sum(normalized_counters[8:9]) == 0:
		hyperonym_feature = 0
	else:
		hyperonym_feature = sum(normalized_counters[8:9])/2

	other_feature = normalized_counters[10]

	return antonym_feature, synonym_feature, hyperonym_feature, other_feature

def compute_semantic_relations(pipe_lemmas, number_of_sources=3):
	""" Function used to compute the semantic relations coefficient between two sentences in the corpus """

	relations_file_path = os.path.join('semantic_relations', 'triplos_10recs', 'triplos_todos_10recs_n.txt')

	with open(relations_file_path) as relations_file:
		relations_data = relations_file.read().splitlines()

	relations_file.close()

	#there is a problem with rstrip, try to fix it in order to remove the tab
	relations_data_strip = [line.replace('\t', ' ') for line in relations_data]
	relations_data_split = [[relation] for relation in relations_data_strip]
	relations_data_string_split = [relation[0].split() for relation in relations_data_split]

	#keep only the semantic relations that were extracted from more than number_of_sources sources
	relations_sources = [relation for relation in relations_data_string_split if int(relation[3]) >= number_of_sources]

	antonym_strings = ["ANTONIMO_N_DE", "ANTONIMO_V_DE", "ANTONIMO_ADJ_DE", "ANTONIMO_ADV_DE"]
	synonym_strings = ["SINONIMO_N_DE", "SINONIMO_V_DE", "SINONIMO_ADJ_DE", "SINONIMO_ADV_DE"]
	hyperonym_strings = ["HIPERONIMO_DE", "HIPERONIMO_ACCAO_DE"]

	antonimo_n_de = collections.defaultdict(list)
	antonimo_v_de = collections.defaultdict(list)
	antonimo_adj_de = collections.defaultdict(list)
	antonimo_adv_de = collections.defaultdict(list)

	sinonimo_n_de = collections.defaultdict(list)
	sinonimo_v_de = collections.defaultdict(list)
	sinonimo_adj_de = collections.defaultdict(list)
	sinonimo_adv_de = collections.defaultdict(list)

	hiperonimo_de = collections.defaultdict(list)
	hiperonimo_accao_de = collections.defaultdict(list)

	outra = collections.defaultdict(list)

	for relation in relations_sources:
		if relation[1] == antonym_strings[0]:
			antonimo_n_de[relation[0]].append(relation[2])
		elif relation[1] == antonym_strings[1]:
			antonimo_v_de[relation[0]].append(relation[2])
		elif relation[1] == antonym_strings[2]:
			antonimo_adj_de[relation[0]].append(relation[2])
		elif relation[1] == antonym_strings[3]:
			antonimo_adv_de[relation[0]].append(relation[2])
		elif relation[1] == synonym_strings[0]:
			sinonimo_n_de[relation[0]].append(relation[2])
		elif relation[1] == synonym_strings[1]:
			sinonimo_v_de[relation[0]].append(relation[2])
		elif relation[1] == synonym_strings[2]:
			sinonimo_adj_de[relation[0]].append(relation[2])
		elif relation[1] == synonym_strings[3]:
			sinonimo_adv_de[relation[0]].append(relation[2])
		elif relation[1] == hyperonym_strings[0]:
			hiperonimo_de[relation[0]].append(relation[2])
		elif relation[1] == hyperonym_strings[1]:
			hiperonimo_accao_de[relation[0]].append(relation[2])
		else:
			outra[relation[0]].append(relation[2])

	dictionaries_relations_list = []

	dictionaries_relations_list.append(antonimo_n_de)
	dictionaries_relations_list.append(antonimo_v_de)
	dictionaries_relations_list.append(antonimo_adj_de)
	dictionaries_relations_list.append(antonimo_adv_de)
	dictionaries_relations_list.append(sinonimo_n_de)
	dictionaries_relations_list.append(sinonimo_v_de)
	dictionaries_relations_list.append(sinonimo_adj_de)
	dictionaries_relations_list.append(sinonimo_adv_de)
	dictionaries_relations_list.append(hiperonimo_de)
	dictionaries_relations_list.append(hiperonimo_accao_de)
	dictionaries_relations_list.append(outra)

	preprocessed_lemmatized_corpus = preprocessing(pipe_lemmas, 1, 1, 0, 0)

	# create only one semantic feature
	#preprocessed_lemmatized_corpus['semantics'] = preprocessed_lemmatized_corpus.apply(lambda pair: semantic_relations_coefficient(pair['text'], pair['response'], dictionaries_relations_list), axis=1)

	# create one semantic feature for each semantic relation
	preprocessed_lemmatized_corpus[['antonyms', 'synonyms', 'hyperonyms', 'other']] = preprocessed_lemmatized_corpus.apply(lambda pair: pd.Series(semantic_relations_coefficient(pair['text'], pair['response'], dictionaries_relations_list)), axis=1)

	return preprocessed_lemmatized_corpus
