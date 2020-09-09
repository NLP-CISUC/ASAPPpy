'''
Module used for feature extraction of a corpus.
'''

import time
import os
import collections
import argparse
import spacy
import numpy as np
from gensim.models import KeyedVectors
from xml.etree import cElementTree as ET
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

from NLPyPort.FullPipeline import new_full_pipe

from ASAPPpy import ROOT_PATH

from .scripts.xml_reader import read_xml
from .scripts.xml_reader import read_xml_no_attributes

from .models.word2vec.word2vec import word2vec_model
from .models.fastText.fasttext import fasttext_model
from .models.ontoPT.ptlkb import ptlkb_model
from .load_embeddings import word_embeddings_model
from .scripts.tools import preprocessing, compute_tfidf_matrix, read_corpus, write_features_to_csv

from .feature_engineering.lexical_features import create_word_ngrams, create_multiple_word_ngrams, create_character_ngrams, create_multiple_character_ngrams, compute_jaccard, compute_dice, compute_overlap, NG
from .feature_engineering.syntactic_features import compute_pos, dependency_parsing
from .feature_engineering.semantic_features import compute_ner, compute_semantic_relations

def build_sentences_from_tokens(tokens):
	""" Function used to rebuild the sentences from the tokens returned by the pipeline """

	sentences = []
	tmp_sentence = []

	for elem in tokens:
		if elem == "EOS":
			tmp_sentence = ' '.join(tmp_sentence)
			sentences.append(tmp_sentence)
			tmp_sentence = []
		else:
			tmp_sentence.append(elem)

	return sentences

def extract_features(run_pipeline, corpus, preprocessed_corpus, word2vec_mdl=None, fasttext_mdl=None, ptlkb64_mdl=None, glove300_mdl=None, numberbatch_mdl=None, f_selection=None):
	""" Function used to extract the features """

	# run NLPyPort pipeline before extracting the features
	if run_pipeline == 1:
		# if system_mode == 0:
		# 	corpus_path = os.path.join('datasets', 'FAQ_todas_variantes_texto_clean.txt')
		# elif system_mode == 1:
		# 	corpus_path = os.path.join('NLPyPort', 'SampleInput', 'train_corpus_ptpt_ptbr.txt')
		# elif system_mode == 2:
		# 	corpus_path = os.path.join('NLPyPort', 'SampleInput', 'train_test_corpus_ptpt_ptbr.txt')
		# elif system_mode == 3:
		# 	corpus_path = "tmp_file.txt"
		# elif system_mode == 4:
		# 	corpus_path = os.path.join('NLPyPort', 'SampleInput', 'complete_assin_training_corpus.txt')
		# elif system_mode == 5:
		# 	corpus_path = os.path.join('NLPyPort', 'SampleInput', 'complete_assin_training_assin1_testing_corpus.txt')

		start_time = time.time()
		print("Started running the pipeline")

		pipeline_output = new_full_pipe(corpus, options={"string_or_array":True})

		# there's still a bug in the pipeline that makes it output two EOS tokens at the end of each run, reason why we read the output to the penultimate token.
		tags = build_sentences_from_tokens(pipeline_output.pos_tags)
		lemmas = build_sentences_from_tokens(pipeline_output.lemas)
		entities = build_sentences_from_tokens(pipeline_output.entities)

		print("Finished running the pipeline successfully")
		print("--- %s seconds ---" %(time.time() - start_time))
		print('\a')

	features = []

	# create word ngrams of different sizes
	if (f_selection is None) or (1 in f_selection[0:9:3]):
		word_ngrams_1 = create_word_ngrams(preprocessed_corpus, 1)

	if (f_selection is None) or (1 in f_selection[1:9:3]):
		word_ngrams_2 = create_word_ngrams(preprocessed_corpus, 2)

	if (f_selection is None) or (1 in f_selection[2:9:3]):
		word_ngrams_3 = create_word_ngrams(preprocessed_corpus, 3)

	# compute distance coefficients for these ngrams
	if (f_selection is None) or f_selection[0]:
		wn_jaccard_1 = compute_jaccard(word_ngrams_1)
	if (f_selection is None) or f_selection[1]:
		wn_jaccard_2 = compute_jaccard(word_ngrams_2)
	if (f_selection is None) or f_selection[2]:
		wn_jaccard_3 = compute_jaccard(word_ngrams_3)

	if (f_selection is None) or f_selection[3]:
		wn_dice_1 = compute_dice(word_ngrams_1)
	if (f_selection is None) or f_selection[4]:
		wn_dice_2 = compute_dice(word_ngrams_2)
	if (f_selection is None) or f_selection[5]:
		wn_dice_3 = compute_dice(word_ngrams_3)

	if (f_selection is None) or f_selection[6]:
		wn_overlap_1 = compute_overlap(word_ngrams_1)
	if (f_selection is None) or f_selection[7]:
		wn_overlap_2 = compute_overlap(word_ngrams_2)
	if (f_selection is None) or f_selection[8]:
		wn_overlap_3 = compute_overlap(word_ngrams_3)

	# create character ngrams of different sizes
	if (f_selection is None) or (1 in f_selection[9:18:3]):
		character_ngrams_2 = create_character_ngrams(preprocessed_corpus, 2)
	if (f_selection is None) or (1 in f_selection[10:18:3]):
		character_ngrams_3 = create_character_ngrams(preprocessed_corpus, 3)
	if (f_selection is None) or (1 in f_selection[11:18:3]):
		character_ngrams_4 = create_character_ngrams(preprocessed_corpus, 4)

	# compute distance coefficients for these ngrams
	if (f_selection is None) or f_selection[9]:
		cn_jaccard_2 = compute_jaccard(character_ngrams_2)
	if (f_selection is None) or f_selection[10]:
		cn_jaccard_3 = compute_jaccard(character_ngrams_3)
	if (f_selection is None) or f_selection[11]:
		cn_jaccard_4 = compute_jaccard(character_ngrams_4)

	if (f_selection is None) or f_selection[12]:
		cn_dice_2 = compute_dice(character_ngrams_2)
	if (f_selection is None) or f_selection[13]:
		cn_dice_3 = compute_dice(character_ngrams_3)
	if (f_selection is None) or f_selection[14]:
		cn_dice_4 = compute_dice(character_ngrams_4)

	if (f_selection is None) or f_selection[15]:
		cn_overlap_2 = compute_overlap(character_ngrams_2)
	if (f_selection is None) or f_selection[16]:
		cn_overlap_3 = compute_overlap(character_ngrams_3)
	if (f_selection is None) or f_selection[17]:
		cn_overlap_4 = compute_overlap(character_ngrams_4)

	if word2vec_mdl:
		if (f_selection is None) or f_selection[18]:
			word2vec = word2vec_model(word2vec_mdl, corpus, 0, 1, 0)
		if (f_selection is None) or f_selection[19]:
			word2vec_tfidf = word2vec_model(word2vec_mdl, corpus, 1, 1, 0)

	if fasttext_mdl:
		if (f_selection is None) or f_selection[20]:
			fasttext = fasttext_model(fasttext_mdl, corpus, 0, 1, 0)
		if (f_selection is None) or f_selection[21]:
			fasttext_tfidf = fasttext_model(fasttext_mdl, corpus, 1, 1, 0)

	if ptlkb64_mdl:
		if (f_selection is None) or f_selection[22]:
			# if run_pipeline == 0:
			# 	ptlkb_64 = ptlkb.word_embeddings_model(run_pipeline, system_mode, ptlkb64_mdl, 0, 1, 0)
			# else:
			ptlkb_64 = ptlkb_model(ptlkb64_mdl, 0, 1, 0, lemmas)
		if (f_selection is None) or f_selection[23]:
			# if run_pipeline == 0:
			# 	ptlkb_64_tfidf = ptlkb.word_embeddings_model(run_pipeline, system_mode, ptlkb64_mdl, 1, 1, 0)
			# else:
			ptlkb_64_tfidf = ptlkb_model(ptlkb64_mdl, 1, 1, 0, lemmas)

	if glove300_mdl:
		if (f_selection is None) or f_selection[24]:
			glove_300 = word_embeddings_model(glove300_mdl, corpus, 0, 1, 0)
		if (f_selection is None) or f_selection[25]:
			glove_300_tfidf = word_embeddings_model(glove300_mdl, corpus, 1, 1, 0)

	# compute tfidf matrix - padding was applied to vectors of different sizes by adding zeros to the smaller vector of the pair
	if (f_selection is None) or f_selection[26]:
		tfidf_corpus = preprocessing(corpus, 0, 0, 0, 1)
		tfidf_matrix = compute_tfidf_matrix(tfidf_corpus, 0, 0, 1)

	# compute semantic relations coefficients
	if (f_selection is None) or (1 in f_selection[27:31]):
		# relations_file_path = os.path.join('semantic_relations', 'triplos_10recs', 'triplos_todos_10recs_n.txt')
		# if run_pipeline == 0:
		# 	semantic_relations = compute_semantic_relations(run_pipeline, system_mode, relations_file_path, 3)
		# else:
		semantic_relations = compute_semantic_relations(lemmas)

	# compute POS tags
	if (f_selection is None) or (1 in f_selection[31:50]):
		# if run_pipeline == 0:
		# 	pos_tags = compute_pos(run_pipeline, system_mode)
		# else:
		pos_tags = compute_pos(tags)

	# compute NERs
	if (f_selection is None) or (1 in f_selection[50:61]):
		# if run_pipeline == 0:
		# 	ners = compute_ner(run_pipeline, system_mode)
		# else:
		ners = compute_ner(entities)

	# compute Syntactic Dependency parsing
	if (f_selection is None) or f_selection[61]:
		dependencies = dependency_parsing(corpus)

	# create multiple word ngrams of different sizes
	if (f_selection is None) or f_selection[62]:
		word_ngrams_1_2 = create_multiple_word_ngrams(preprocessed_corpus, 1, 2)

	# compute the cosine similarity between the multiple word ngrams converted sentences
	if (f_selection is None) or f_selection[62]:
		wn_cosine_1_2 = NG(word_ngrams_1_2)

	# create multiple character ngrams of different sizes
	if (f_selection is None) or f_selection[63]:
		character_ngrams_1_2_3 = create_multiple_character_ngrams(preprocessed_corpus, 1, 2, 3)

	# compute the cosine similarity between the multiple character ngrams converted sentences
	if (f_selection is None) or f_selection[63]:
		cn_cosine_1_2_3 = NG(character_ngrams_1_2_3)

	if numberbatch_mdl:
		if (f_selection is None) or f_selection[64]:
			numberbatch = word_embeddings_model(numberbatch_mdl, corpus, 0, 1, 0)
		if (f_selection is None) or f_selection[65]:
			numberbatch_tfidf = word_embeddings_model(numberbatch_mdl, corpus, 1, 1, 0)

	for pair in range(len(preprocessed_corpus)):
		if f_selection is not None:
			features_pair = []

			if f_selection[0]:
				features_pair.append(wn_jaccard_1['jaccard'][pair])
			if f_selection[1]:
				features_pair.append(wn_jaccard_2['jaccard'][pair])
			if f_selection[2]:
				features_pair.append(wn_jaccard_3['jaccard'][pair])
			if f_selection[3]:
				features_pair.append(wn_dice_1['dice'][pair])
			if f_selection[4]:
				features_pair.append(wn_dice_2['dice'][pair])
			if f_selection[5]:
				features_pair.append(wn_dice_3['dice'][pair])
			if f_selection[6]:
				features_pair.append(wn_overlap_1['overlap'][pair])
			if f_selection[7]:
				features_pair.append(wn_overlap_2['overlap'][pair])
			if f_selection[8]:
				features_pair.append(wn_overlap_3['overlap'][pair])
			if f_selection[9]:
				features_pair.append(cn_jaccard_2['jaccard'][pair])
			if f_selection[10]:
				features_pair.append(cn_jaccard_3['jaccard'][pair])
			if f_selection[11]:
				features_pair.append(cn_jaccard_4['jaccard'][pair])
			if f_selection[12]:
				features_pair.append(cn_dice_2['dice'][pair])
			if f_selection[13]:
				features_pair.append(cn_dice_3['dice'][pair])
			if f_selection[14]:
				features_pair.append(cn_dice_4['dice'][pair])
			if f_selection[15]:
				features_pair.append(cn_overlap_2['overlap'][pair])
			if f_selection[16]:
				features_pair.append(cn_overlap_3['overlap'][pair])
			if f_selection[17]:
				features_pair.append(cn_overlap_4['overlap'][pair])
			if f_selection[18]:
				features_pair.append(word2vec[pair])
			if f_selection[19]:
				features_pair.append(word2vec_tfidf[pair])
			if f_selection[20]:
				features_pair.append(fasttext[pair])
			if f_selection[21]:
				features_pair.append(fasttext_tfidf[pair])
			if f_selection[22]:
				features_pair.append(ptlkb_64[pair])
			if f_selection[23]:
				features_pair.append(ptlkb_64_tfidf[pair])
			if f_selection[24]:
				features_pair.append(glove_300[pair])
			if f_selection[25]:
				features_pair.append(glove_300_tfidf[pair])
			if f_selection[26]:
				features_pair.append(tfidf_matrix[pair])
			if f_selection[27]:
				features_pair.append(semantic_relations['antonyms'][pair])
			if f_selection[28]:
				features_pair.append(semantic_relations['synonyms'][pair])
			if f_selection[29]:
				features_pair.append(semantic_relations['hyperonyms'][pair])
			if f_selection[30]:
				features_pair.append(semantic_relations['other'][pair])
			if f_selection[31] and ('n' in pos_tags.columns):
				features_pair.append(pos_tags['n'][pair])
			if f_selection[32] and ('prop' in pos_tags.columns):
				features_pair.append(pos_tags['prop'][pair])
			if f_selection[33] and ('adj' in pos_tags.columns):
				features_pair.append(pos_tags['adj'][pair])
			if f_selection[34] and ('n-adj' in pos_tags.columns):
				features_pair.append(pos_tags['n-adj'][pair])
			if f_selection[35] and ('v-fin' in pos_tags.columns):
				features_pair.append(pos_tags['v-fin'][pair])
			if f_selection[36] and ('v-inf' in pos_tags.columns):
				features_pair.append(pos_tags['v-inf'][pair])
			if f_selection[37] and ('v-pcp' in pos_tags.columns):
				features_pair.append(pos_tags['v-pcp'][pair])
			if f_selection[38] and ('v-ger' in pos_tags.columns):
				features_pair.append(pos_tags['v-ger'][pair])
			if f_selection[39] and ('art' in pos_tags.columns):
				features_pair.append(pos_tags['art'][pair])
			if f_selection[40] and ('pron-pers' in pos_tags.columns):
				features_pair.append(pos_tags['pron-pers'][pair])
			if f_selection[41] and ('pron-det' in pos_tags.columns):
				features_pair.append(pos_tags['pron-det'][pair])
			if f_selection[42] and ('pron-indp' in pos_tags.columns):
				features_pair.append(pos_tags['pron-indp'][pair])
			if f_selection[43] and ('adv' in pos_tags.columns):
				features_pair.append(pos_tags['adv'][pair])
			if f_selection[44] and ('num' in pos_tags.columns):
				features_pair.append(pos_tags['num'][pair])
			if f_selection[45] and ('prp' in pos_tags.columns):
				features_pair.append(pos_tags['prp'][pair])
			if f_selection[46] and ('intj' in pos_tags.columns):
				features_pair.append(pos_tags['intj'][pair])
			if f_selection[47] and ('conj-s' in pos_tags.columns):
				features_pair.append(pos_tags['conj-s'][pair])
			if f_selection[48] and ('conj-c' in pos_tags.columns):
				features_pair.append(pos_tags['conj-c'][pair])
			if f_selection[49] and ('punc' in pos_tags.columns):
				features_pair.append(pos_tags['punc'][pair])
			if f_selection[50] and ('all_ners' in ners.columns):
				features_pair.append(ners['all_ners'][pair])
			if f_selection[51] and ('B-ABSTRACCAO' in ners.columns):
				features_pair.append(ners['B-ABSTRACCAO'][pair])
			if f_selection[52] and ('B-ACONTECIMENTO' in ners.columns):
				features_pair.append(ners['B-ACONTECIMENTO'][pair])
			if f_selection[53] and ('B-COISA' in ners.columns):
				features_pair.append(ners['B-COISA'][pair])
			if f_selection[54] and ('B-LOCAL' in ners.columns):
				features_pair.append(ners['B-LOCAL'][pair])
			if f_selection[55] and ('B-OBRA' in ners.columns):
				features_pair.append(ners['B-OBRA'][pair])
			if f_selection[56] and ('B-ORGANIZACAO' in ners.columns):
				features_pair.append(ners['B-ORGANIZACAO'][pair])
			if f_selection[57] and ('B-OUTRO' in ners.columns):
				features_pair.append(ners['B-OUTRO'][pair])
			if f_selection[58] and ('B-PESSOA' in ners.columns):
				features_pair.append(ners['B-PESSOA'][pair])
			if f_selection[59] and ('B-TEMPO' in ners.columns):
				features_pair.append(ners['B-TEMPO'][pair])
			if f_selection[60] and ('B-VALOR' in ners.columns):
				features_pair.append(ners['B-VALOR'][pair])
			if f_selection[61]:
				features_pair.append(dependencies['dependency_parsing_jc'][pair])
			if f_selection[62]:
				features_pair.append(wn_cosine_1_2['NG'][pair])
			if f_selection[63]:
				features_pair.append(cn_cosine_1_2_3['NG'][pair])
			if f_selection[64]:
				features_pair.append(numberbatch[pair])
			if f_selection[65]:
				features_pair.append(numberbatch_tfidf[pair])

			tuple_features_pair = tuple(features_pair)
			features.append(tuple_features_pair)

		else:
			flag = 1
			features_pair = []
			used_features = [False] * 66

			if flag == 1:
				used_features[0] = True
				features_pair.append(wn_jaccard_1['jaccard'][pair])
			if flag == 1:
				used_features[1] = True
				features_pair.append(wn_jaccard_2['jaccard'][pair])
			if flag == 1:
				used_features[2] = True
				features_pair.append(wn_jaccard_3['jaccard'][pair])
			if flag == 1:
				used_features[3] = True
				features_pair.append(wn_dice_1['dice'][pair])
			if flag == 1:
				used_features[4] = True
				features_pair.append(wn_dice_2['dice'][pair])
			if flag == 1:
				used_features[5] = True
				features_pair.append(wn_dice_3['dice'][pair])
			if flag == 1:
				used_features[6] = True
				features_pair.append(wn_overlap_1['overlap'][pair])
			if flag == 1:
				used_features[7] = True
				features_pair.append(wn_overlap_2['overlap'][pair])
			if flag == 1:
				used_features[8] = True
				features_pair.append(wn_overlap_3['overlap'][pair])
			if flag == 1:
				used_features[9] = True
				features_pair.append(cn_jaccard_2['jaccard'][pair])
			if flag == 1:
				used_features[10] = True
				features_pair.append(cn_jaccard_3['jaccard'][pair])
			if flag == 1:
				used_features[11] = True
				features_pair.append(cn_jaccard_4['jaccard'][pair])
			if flag == 1:
				used_features[12] = True
				features_pair.append(cn_dice_2['dice'][pair])
			if flag == 1:
				used_features[13] = True
				features_pair.append(cn_dice_3['dice'][pair])
			if flag == 1:
				used_features[14] = True
				features_pair.append(cn_dice_4['dice'][pair])
			if flag == 1:
				used_features[15] = True
				features_pair.append(cn_overlap_2['overlap'][pair])
			if flag == 1:
				used_features[16] = True
				features_pair.append(cn_overlap_3['overlap'][pair])
			if flag == 1:
				used_features[17] = True
				features_pair.append(cn_overlap_4['overlap'][pair])
			if flag == 0:
				used_features[18] = True
				features_pair.append(word2vec[pair])
			if flag == 0:
				used_features[19] = True
				features_pair.append(word2vec_tfidf[pair])
			if flag == 1:
				used_features[20] = True
				features_pair.append(fasttext[pair])
			if flag == 1:
				used_features[21] = True
				features_pair.append(fasttext_tfidf[pair])
			if flag == 1:
				used_features[22] = True
				features_pair.append(ptlkb_64[pair])
			if flag == 1:
				used_features[23] = True
				features_pair.append(ptlkb_64_tfidf[pair])
			if flag == 1:
				used_features[24] = True
				features_pair.append(glove_300[pair])
			if flag == 1:
				used_features[25] = True
				features_pair.append(glove_300_tfidf[pair])
			if flag == 0:
				used_features[26] = True
				features_pair.append(tfidf_matrix[pair])
			if flag == 0:
				used_features[27] = True
				features_pair.append(semantic_relations['antonyms'][pair])
			if flag == 1:
				used_features[28] = True
				features_pair.append(semantic_relations['synonyms'][pair])
			if flag == 1:
				used_features[29] = True
				features_pair.append(semantic_relations['hyperonyms'][pair])
			if flag == 0:
				used_features[30] = True
				features_pair.append(semantic_relations['other'][pair])
			if flag == 1 and ('n' in pos_tags.columns):
				used_features[31] = True
				features_pair.append(pos_tags['n'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('prop' in pos_tags.columns):
				used_features[32] = True
				features_pair.append(pos_tags['prop'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('adj' in pos_tags.columns):
				used_features[33] = True
				features_pair.append(pos_tags['adj'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('n-adj' in pos_tags.columns):
				used_features[34] = True
				features_pair.append(pos_tags['n-adj'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('v-fin' in pos_tags.columns):
				used_features[35] = True
				features_pair.append(pos_tags['v-fin'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('v-inf' in pos_tags.columns):
				used_features[36] = True
				features_pair.append(pos_tags['v-inf'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('v-pcp' in pos_tags.columns):
				used_features[37] = True
				features_pair.append(pos_tags['v-pcp'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('v-ger' in pos_tags.columns):
				used_features[38] = True
				features_pair.append(pos_tags['v-ger'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('art' in pos_tags.columns):
				used_features[39] = True
				features_pair.append(pos_tags['art'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('pron-pers' in pos_tags.columns):
				used_features[40] = True
				features_pair.append(pos_tags['pron-pers'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('pron-det' in pos_tags.columns):
				used_features[41] = True
				features_pair.append(pos_tags['pron-det'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('pron-indp' in pos_tags.columns):
				used_features[42] = True
				features_pair.append(pos_tags['pron-indp'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('adv' in pos_tags.columns):
				used_features[43] = True
				features_pair.append(pos_tags['adv'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('num' in pos_tags.columns):
				used_features[44] = True
				features_pair.append(pos_tags['num'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('prp' in pos_tags.columns):
				used_features[45] = True
				features_pair.append(pos_tags['prp'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('intj' in pos_tags.columns):
				used_features[46] = True
				features_pair.append(pos_tags['intj'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('conj-s' in pos_tags.columns):
				used_features[47] = True
				features_pair.append(pos_tags['conj-s'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('conj-c' in pos_tags.columns):
				used_features[48] = True
				features_pair.append(pos_tags['conj-c'][pair])
			else:
				features_pair.append(0)
			if flag == 1 and ('punc' in pos_tags.columns):
				used_features[49] = True
				features_pair.append(pos_tags['punc'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('all_ners' in ners.columns):
				used_features[50] = True
				features_pair.append(ners['all_ners'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('B-ABSTRACCAO' in ners.columns):
				used_features[51] = True
				features_pair.append(ners['B-ABSTRACCAO'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('B-ACONTECIMENTO' in ners.columns):
				used_features[52] = True
				features_pair.append(ners['B-ACONTECIMENTO'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('B-COISA' in ners.columns):
				used_features[53] = True
				features_pair.append(ners['B-COISA'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('B-LOCAL' in ners.columns):
				used_features[54] = True
				features_pair.append(ners['B-LOCAL'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('B-OBRA' in ners.columns):
				used_features[55] = True
				features_pair.append(ners['B-OBRA'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('B-ORGANIZACAO' in ners.columns):
				used_features[56] = True
				features_pair.append(ners['B-ORGANIZACAO'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('B-OUTRO' in ners.columns):
				used_features[57] = True
				features_pair.append(ners['B-OUTRO'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('B-PESSOA' in ners.columns):
				used_features[58] = True
				features_pair.append(ners['B-PESSOA'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('B-TEMPO' in ners.columns):
				used_features[59] = True
				features_pair.append(ners['B-TEMPO'][pair])
			else:
				features_pair.append(0)
			if flag == 0 and ('B-VALOR' in ners.columns):
				used_features[60] = True
				features_pair.append(ners['B-VALOR'][pair])
			else:
				features_pair.append(0)
			if flag == 1:
				used_features[61] = True
				features_pair.append(dependencies['dependency_parsing_jc'][pair])
			if flag == 1:
				used_features[62] = True
				features_pair.append(wn_cosine_1_2['NG'][pair])
			if flag == 1:
				used_features[63] = True
				features_pair.append(cn_cosine_1_2_3['NG'][pair])
			if flag == 0:
				used_features[64] = True
				features_pair.append(numberbatch[pair])
			if flag == 0:
				used_features[65] = True
				features_pair.append(numberbatch_tfidf[pair])

			tuple_features_pair = tuple(features_pair)
			features.append(tuple_features_pair)

	if f_selection is not None:
		return np.array(features)
	else:
		return np.array(features), used_features

def debug_data(data, filename):
	""" Function used to debug the corpus state during preprocessing """
	if isinstance(data, pd.DataFrame):
		data_to_print = data.values.tolist()
	else:
		data_to_print = data

	with open(filename, 'w') as f:
		for item in data_to_print:
			f.write("%s\n" % item)

def load_embeddings_models():
	""" Function used to load the word-embedding models """

	# ---LOADING WORD2VEC MODEL---
	model_load_path = os.path.join(ROOT_PATH, 'models', 'word2vec', 'NILC', 'nilc_cbow_s300_300k.txt')
	# model_load_path = os.path.join('models', 'word2vec', 'NILC', 'nilc_skip_s300.txt')
	start_time = time.time()
	print("Started loading the word2vec model")
	word2vec_model = KeyedVectors.load_word2vec_format(model_load_path)
	# word2vec_model = None
	print("Model loaded")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')

	# ---LOADING FASTTEXT MODEL---
	model_path = os.path.join(ROOT_PATH, 'models', 'fastText', 'cc.pt.300_300k.vec')
	start_time = time.time()
	print("Started loading the fasttext model")
	fasttext_model = KeyedVectors.load_word2vec_format(model_path)
	# fasttext_model = None
	print("Model loaded")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')	

	# ---LOADING PT-LKB MODEL---
	model_load_path = os.path.join(ROOT_PATH, 'models', 'ontoPT', 'PT-LKB_embeddings_64', 'ptlkb_64_30_200_p_str.emb')
	# model_load_path = os.path.join('models', 'ontoPT', 'PT-LKB_embeddings_128', 'ptlkb_128_80_10_p_str.emb')
	start_time = time.time()
	print("Started loading the PT-LKB-64 model")
	ptlkb64_model = KeyedVectors.load_word2vec_format(model_load_path)
	# ptlkb64_model = None
	print("Model loaded")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')

	# ---LOADING GLOVE-300 MODEL---
	model_load_path = os.path.join(ROOT_PATH, 'models', 'glove', 'glove_s300_300k.txt')
	# model_load_path = os.path.join('models', 'glove', 'glove_s100.txt')
	start_time = time.time()
	print("Started loading the GLOVE 300 dimensions model")
	glove300_model = KeyedVectors.load_word2vec_format(model_load_path)
	# glove300_model = None
	print("Model loaded")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')

	# ---LOADING NUMBERBATCH MODEL---
	model_load_path = os.path.join(ROOT_PATH, 'models', 'numberbatch', 'numberbatch-17.02_pt_tratado.txt')
	start_time = time.time()
	print("Started loading the NUMBERBATCH dimensions model")
	numberbatch_model = KeyedVectors.load_word2vec_format(model_load_path)
	# numberbatch_model = None
	print("Model loaded")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')

	return word2vec_model, fasttext_model, ptlkb64_model, glove300_model, numberbatch_model

def best_percentile_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor):
	""" Function used to select the best percentile selector """
	percentile_score = 0
	percentiles = [25, 35, 45, 50, 55, 65, 75]
	# percentiles = [45]
	percentile_selector = None
	percentile_train_features_selected = None
	percentile_test_features_selected = None

	for percentile in percentiles:
		print(percentile)
		temp_percentile_selector = SelectPercentile(score_func=f_regression, percentile=percentile)
		temp_percentile_selector.fit(train_features, train_similarity_target)
		temp_percentile_train_features_selected = temp_percentile_selector.transform(train_features)
		temp_percentile_test_features_selected = temp_percentile_selector.transform(test_features)

		regressor.fit(temp_percentile_train_features_selected, train_similarity_target)

		temp_score = regressor.score(temp_percentile_test_features_selected, test_similarity_target)
		print("The score on the selected features (Percentile Selector): %.3f" % temp_score)

		if temp_score > percentile_score:
			percentile_score = temp_score
			percentile_selector = temp_percentile_selector
			percentile_train_features_selected = temp_percentile_train_features_selected
			percentile_test_features_selected = temp_percentile_test_features_selected

	percentile_mask = percentile_selector.get_support()
	print("This is the percentile mask: ")
	print(percentile_mask)

	return percentile_selector, percentile_score, percentile_train_features_selected, percentile_test_features_selected, percentile_mask

def best_model_based_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor):
	""" Function used to select the best model based selector """
	model_based_score = 0
	scaling_factors = ["0.25*mean", "0.5*mean", "median", "1.25*mean", "1.5*mean"]
	# scaling_factors = ["0.5*mean", "median"]
	model_based_selector = None
	model_based_train_features_selected = None
	model_based_test_features_selected = None

	for factor in scaling_factors:
		print(factor)
		temp_model_based_selector = SelectFromModel(RandomForestRegressor(n_estimators=100), threshold=factor)
		temp_model_based_selector.fit(train_features, train_similarity_target)
		temp_model_based_train_features_selected = temp_model_based_selector.transform(train_features)
		temp_model_based_test_features_selected = temp_model_based_selector.transform(test_features)

		regressor.fit(temp_model_based_train_features_selected, train_similarity_target)

		temp_score = regressor.score(temp_model_based_test_features_selected, test_similarity_target)
		print("The score on the selected features (Model Based Selector): %.3f" % temp_score)

		if temp_score > model_based_score:
			model_based_score = temp_score
			model_based_selector = temp_model_based_selector
			model_based_train_features_selected = temp_model_based_train_features_selected
			model_based_test_features_selected = temp_model_based_test_features_selected

	model_based_mask = model_based_selector.get_support()
	print("This is the model based mask: ")
	print(model_based_mask)

	return model_based_selector, model_based_score, model_based_train_features_selected, model_based_test_features_selected, model_based_mask

def best_iterative_based_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor):
	""" Function used to select the best iterative based selector """
	iterative_based_score = 0
	# given that all pairs use the same amount of features, the position 0 was arbitrarily selected to compute the number of features being used
	min_number_features = int(0.15*len(train_features[0]))
	max_number_features = int(0.85*len(train_features[0]))

	# min_number_features = 19
	# max_number_features = 20

	iterative_based_selector = None
	iterative_based_train_features_selected = None
	iterative_based_test_features_selected = None

	for i in range(min_number_features, max_number_features):
		print(i)
		temp_iterative_based_selector = RFE(RandomForestRegressor(n_estimators=100), n_features_to_select=i)
		temp_iterative_based_selector.fit(train_features, train_similarity_target)
		temp_iterative_based_train_features_selected = temp_iterative_based_selector.transform(train_features)
		temp_iterative_based_test_features_selected = temp_iterative_based_selector.transform(test_features)

		regressor.fit(temp_iterative_based_train_features_selected, train_similarity_target)

		temp_score = regressor.score(temp_iterative_based_test_features_selected, test_similarity_target)
		print("The score on the selected features (Iterative Based Selector): %.3f" % temp_score)

		if temp_score > iterative_based_score:
			iterative_based_score = temp_score
			iterative_based_selector = temp_iterative_based_selector
			iterative_based_train_features_selected = temp_iterative_based_train_features_selected
			iterative_based_test_features_selected = temp_iterative_based_test_features_selected

	iterative_based_mask = iterative_based_selector.get_support()
	print("This is the iterative based mask: ")
	print(iterative_based_mask)

	return iterative_based_selector, iterative_based_score, iterative_based_train_features_selected, iterative_based_test_features_selected, iterative_based_mask

def rfe_cross_validation(train_features, train_similarity_target, test_features):
	estimator = GradientBoostingRegressor(n_estimators=100)
	rfecv = RFECV(estimator, step=1, cv=10)
	rfecv.fit(train_features, train_similarity_target)

	selected_train_features = rfecv.transform(train_features)
	selected_test_features = rfecv.transform(test_features)

	rfecv_mask = rfecv.get_support()
	print(rfecv_mask)

	return selected_train_features, selected_test_features

def feature_selection(train_features, test_features, train_similarity_target, test_similarity_target, regressor, used_features):
	""" Function used to perform feature selection """
	# percentile selector
	percentile_selector, percentile_score, percentile_train_features_selected, percentile_test_features_selected, percentile_mask = best_percentile_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor)

	# model based selector
	model_based_selector, model_based_score, model_based_train_features_selected, model_based_test_features_selected, model_based_mask = best_model_based_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor)

	# iterative based selector
	iterative_based_selector, iterative_based_score, iterative_based_train_features_selected, iterative_based_test_features_selected, iterative_based_mask = best_iterative_based_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor)

	all_scores = []

	regressor.fit(train_features, train_similarity_target)
	print("The score on all features: %.3f" % regressor.score(test_features, test_similarity_target))
	all_scores.append(regressor.score(test_features, test_similarity_target))

	# show results for the percentile selector
	all_scores.append(percentile_score)

	# show results for the model based selector
	all_scores.append(model_based_score)

	# show results for the iterative based selector
	all_scores.append(iterative_based_score)

	max_value_position = all_scores.index(max(all_scores))

	if max_value_position == 0:
		print("Returning all features!\n")
		return train_features, test_features
	elif max_value_position == 1:
		percentile_mask = build_mask(percentile_mask, used_features)
		mask_save_path = os.path.join('feature_selection_masks', 'percentile_mask.txt')
		debug_data(percentile_mask, mask_save_path)

		print("Returning features selected with the percentile selector!\n")
		return percentile_selector, percentile_train_features_selected, percentile_test_features_selected
	elif max_value_position == 2:
		model_based_mask = build_mask(model_based_mask, used_features)
		mask_save_path = os.path.join('feature_selection_masks', 'model_based_mask.txt')
		debug_data(model_based_mask, mask_save_path)

		print("Returning features selected with the model based selector!\n")
		return model_based_selector, model_based_train_features_selected, model_based_test_features_selected
	else:
		iterative_based_mask = build_mask(iterative_based_mask, used_features)
		mask_save_path = os.path.join('feature_selection_masks', 'iterative_based_mask.txt')
		debug_data(iterative_based_mask, mask_save_path)

		print("Returning features selected with the iterative based selector!\n")
		return iterative_based_selector, iterative_based_train_features_selected, iterative_based_test_features_selected

def build_mask(mask, unused_features_positions):
	""" Function used to complete the mask with unused features not available for feature selection """
	final_mask = mask.tolist()

	for i in range(len(unused_features_positions)):
		if not unused_features_positions[i]:
			final_mask.insert(i, False)

	return final_mask

def aux_best_percentile_selector(train_features, test_features, train_similarity_target, test_similarity_target, regressor, used_features):
	""" Function used to select the best percentile selector """
	percentile_score = 0
	percentiles = [25, 35, 45, 50, 55, 65, 75]
	# percentiles = [45]
	percentile_selector = None
	percentile_train_features_selected = None
	percentile_test_features_selected = None

	for percentile in percentiles:
		print(percentile)
		temp_percentile_selector = SelectPercentile(score_func=f_regression, percentile=percentile)
		temp_percentile_selector.fit(train_features, train_similarity_target)
		temp_percentile_train_features_selected = temp_percentile_selector.transform(train_features)
		temp_percentile_test_features_selected = temp_percentile_selector.transform(test_features)

		regressor.fit(temp_percentile_train_features_selected, train_similarity_target)

		temp_score = regressor.score(temp_percentile_test_features_selected, test_similarity_target)
		print("The score on the selected features (Percentile Selector): %.3f" % temp_score)

		if temp_score > percentile_score:
			percentile_score = temp_score
			percentile_selector = temp_percentile_selector
			percentile_train_features_selected = temp_percentile_train_features_selected
			percentile_test_features_selected = temp_percentile_test_features_selected

	percentile_mask = percentile_selector.get_support()
	print("This is the percentile mask: ")
	print(percentile_mask)

	percentile_mask = build_mask(percentile_mask, used_features)
	mask_save_path = os.path.join('feature_selection_masks', 'assin2_percentile_based_mask.txt')
	debug_data(percentile_mask, mask_save_path)

	return percentile_train_features_selected, percentile_test_features_selected, percentile_selector

def run_feature_extraction(word2vec_model=None, fasttext_model=None, ptlkb64_model=None, glove300_model=None, numberbatch_model=None):
	""" Function used to compute the models """

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('test', help='XML file with test data')
	parser.add_argument('output', help='Output tagged XML file')
	args = parser.parse_args()

	"""
	system_mode = 0 -> uses the variant questions with the system
	system_mode = 1 -> uses the PTPT and PTBR train ASSIN collection datasets with the system
	system_mode = 2 -> uses the PTPT and PTBR train and test ASSIN collection datasets with the system
	system_mode = 3 -> uses the Whoosh collection with the system
	system_mode = 4 -> uses ASSIN 1 and ASSIN 2 training collection datasets with the system
	system_mode = 5 -> uses ASSIN 1 training and testing collection and ASSIN 2 training collection datasets with the system

	run_pipeline = 0 -> uses the pre-computed files with the components needed to extract some features
	run_pipeline = 1 -> uses NLPyPort pipeline which avoids having to pre-compute certain components to extract features
	"""

	system_mode = 5
	run_pipeline = 1

	# Flag to indicate if the extracted features should be written to a file (1) or not (0)
	features_to_file_flag = 0

	# extract labels
	train_pairs = []
	train_pairs.extend(read_xml(ROOT_PATH + "/datasets/assin/assin1/assin-ptpt-train.xml", need_labels=True))
	train_pairs.extend(read_xml(ROOT_PATH + "/datasets/assin/assin1/assin-ptbr-train.xml", need_labels=True))

	if system_mode == 2 or system_mode == 5:
		train_pairs.extend(read_xml(ROOT_PATH + "/datasets/assin/assin1/assin-ptpt-test.xml", need_labels=True))
		train_pairs.extend(read_xml(ROOT_PATH + "/datasets/assin/assin1/assin-ptbr-test.xml", need_labels=True))
	if system_mode == 4 or system_mode == 5:
		train_pairs.extend(read_xml(ROOT_PATH + "/datasets/assin/assin2/assin2-train-only.xml", need_labels=True))

	train_similarity_target = np.array([pair.similarity for pair in train_pairs])

	# extract training features
	train_corpus = read_corpus(train_pairs)

	# debug_data(train_corpus, "finetune.train.raw")
	# print("Wrote training corpus")

	# preprocessing(text, tokenization=0, rm_stopwords=0, numbers_to_text=0, to_tfidf=0)
	preprocessed_train_corpus = preprocessing(train_corpus, 0, 0, 0, 0)
	train_features, used_train_features = extract_features(run_pipeline, train_corpus, preprocessed_train_corpus, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb64_mdl=ptlkb64_model, glove300_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

	# write train features to a .csv file
	if features_to_file_flag == 1:
		write_features_to_csv(train_pairs, train_features, "assin1-train-test-assin2-train-ftrain.csv")

	#############################################################
	test_pairs_dev = read_xml('datasets/assin/assin2/assin2-dev.xml', need_labels=False)

	test_corpus_dev = read_corpus(test_pairs_dev)
	# preprocessing(text, tokenization=0, rm_stopwords=0, numbers_to_text=0, to_tfidf=0)
	preprocessed_test_corpus_dev = preprocessing(test_corpus_dev, 0, 0, 0, 0)
	test_features_dev, used_test_features_dev = extract_features(run_pipeline, test_corpus_dev, preprocessed_test_corpus_dev, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb64_mdl=ptlkb64_model, glove300_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

	test_pairs_selection = read_xml('datasets/assin/assin2/assin2-dev.xml', need_labels=True)
	test_similarity_target = np.array([pair.similarity for pair in test_pairs_selection])
	#############################################################

	# extract test features
	# test_pairs = read_xml(args.test, need_labels=False)

	# uncomment next line and comment previous one to compute ASSIN 2 submission results
	test_pairs = read_xml_no_attributes(args.test)

	test_corpus = read_corpus(test_pairs)
	# preprocessing(text, tokenization=0, rm_stopwords=0, numbers_to_text=0, to_tfidf=0)
	preprocessed_test_corpus = preprocessing(test_corpus, 0, 0, 0, 0)
	test_features, used_test_features = extract_features(run_pipeline, test_corpus, preprocessed_test_corpus, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb64_mdl=ptlkb64_model, glove300_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

	# write test features to a .csv file
	if features_to_file_flag == 1:
		write_features_to_csv(test_pairs, test_features, "assin1-train-test-assin2-train-ftest.csv")

	# extract test features for feature selection (labels needed in order to perform evaluation)
	# test_pairs_selection = read_xml(args.test, need_labels=True)
	# test_similarity_target = np.array([pair.similarity for pair in test_pairs_selection])

	'''
	Select one type of regressor from scikit-learn. Here is a list with some examples: 
		- GaussianProcessRegressor()
		- DecisionTreeRegressor()
		- LinearRegression()
		- BaggingRegressor(n_estimators=100)
		- AdaBoostRegressor(n_estimators=100)
		- GradientBoostingRegressor()
		- RandomForestRegressor(n_estimators=100)
	'''

	regressor = SVR(gamma='scale', C=10.0, kernel='rbf')

	# ensemble = VotingRegressor(estimators=[('svr', regressor_1), ('gb', regressor_2), ('rf', regressor_3)])

	# params = {'svr__C': [1.0, 10.0, 100.0], 'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'rf__n_estimators': [10, 20, 100, 200]}

	# params = {'kernel':('linear', 'poly', 'rbf', 'sigmoid')}

	# regressor = GridSearchCV(regressor_1, params, cv=5)

	use_feature_selection = 0

	if use_feature_selection:
		# selected_selector, selected_train_features, selected_test_features = feature_selection(train_features, test_features_dev, train_similarity_target, test_similarity_target, regressor, used_train_features)
		# selected_train_features, selected_test_features = rfe_cross_validation(train_features, train_similarity_target, test_features)
		selected_train_features, selected_test_features, percentile_selector = aux_best_percentile_selector(train_features, test_features_dev, train_similarity_target, test_similarity_target, regressor, used_train_features)

		test_features_selected = percentile_selector.transform(test_features)
		# test_features_selected = selected_selector.transform(test_features)

		regressor.fit(selected_train_features, train_similarity_target)

		# save model to disk
		model_save_path = os.path.join('trained_models', 'SVR_FS.joblib')
		dump(regressor, model_save_path)

		# apply model to the test dataset
		## this needs to be fixed in order to take advantage of the manual feature selection
		predicted_similarity = regressor.predict(test_features_selected)
		# predicted_similarity = regressor.predict(test_features_selected)
	else:
		regressor.fit(train_features, train_similarity_target)

		# save model to disk
		model_save_path = os.path.join('trained_models', 'SVR_NFS.joblib')
		dump(regressor, model_save_path)

		# apply model to the test dataset
		predicted_similarity = regressor.predict(test_features)

	# write output
	tree = ET.parse(args.test)
	root = tree.getroot()
	for i in range(len(test_pairs)):
		pairs = root[i]
		pairs.set('entailment', "None")
		pairs.set('similarity', str(predicted_similarity[i]))

	tree.write(args.output, 'utf-8')

# if __name__ == '__main__':
# 	system_mode = 4
# 	run_pipeline = 1

# 	# Flag to indicate if the extracted features should be written to a file (1) or not (0)
# 	features_to_file_flag = 0

# 	# extract labels
# 	train_pairs = []
# 	# train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptpt-train.xml", need_labels=True))
# 	# train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptbr-train.xml", need_labels=True))

# 	if system_mode == 2 or system_mode == 5:
# 		# train_pairs.extend(read_xml("assin-ptpt-test.xml", need_labels=True))
# 		train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptbr-test.xml", need_labels=True))
# 	if system_mode == 4 or system_mode == 5:
# 		train_pairs.extend(read_xml("datasets/assin/assin2/assin2-train-only.xml", need_labels=True))

# 	train_similarity_target = np.array([pair.similarity for pair in train_pairs])

# 	# extract training features
# 	train_corpus = tl.read_corpus(train_pairs)

# 	debug_data(train_corpus, "finetune.train.raw")
# 	print("Wrote training corpus")

	# test_pairs = read_xml_no_attributes("datasets/assin/assin2/assin2-blind-test.xml")

	# test_corpus = tl.read_corpus(test_pairs)

	# debug_data(test_corpus, "finetune.test.raw")
	# print("Wrote testing corpus")
