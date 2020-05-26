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
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from NLPyPort.FullPipeline import new_full_pipe, load_congif_to_list
from scripts.xml_reader import read_xml
from scripts.xml_reader import read_xml_no_attributes

import models.word2vec.word2vec as w2c
import models.fastText.fasttext as ftt
import models.ontoPT.ptlkb as ptlkb
import load_embeddings as ld_emb
import scripts.tools as tl

from feature_engineering.lexical_features import create_word_ngrams, create_multiple_word_ngrams, create_character_ngrams, create_multiple_character_ngrams, compute_jaccard, compute_dice, compute_overlap, NG
from feature_engineering.syntactic_features import compute_pos, dependency_parsing
from feature_engineering.semantic_features import compute_ner, compute_semantic_relations
from feature_selection.feature_selection import feature_selection

config_list = load_congif_to_list()

def extract_features(corpus, preprocessed_corpus, word2vec_mdl=None, fasttext_mdl=None, ptlkb64_mdl=None, glove300_mdl=None, numberbatch_mdl=None):
	""" Function used to extract the features """

	# start_time = time.time()
	# print("Started running the pipeline")

	pipeline_output = new_full_pipe(corpus, options={"string_or_array":True}, config_list=config_list)

	# there's still a bug in the pipeline that makes it output two EOS tokens at the end of each run, reason why we read the output to the penultimate token.
	tags = tl.build_sentences_from_tokens(pipeline_output.pos_tags)
	lemmas = tl.build_sentences_from_tokens(pipeline_output.lemas)
	entities = tl.build_sentences_from_tokens(pipeline_output.entities)

	# print("Finished running the pipeline successfully")
	# print("--- %s seconds ---" %(time.time() - start_time))
	# print('\a')

	features = []

	# create word ngrams of different sizes
	word_ngrams_1 = create_word_ngrams(preprocessed_corpus, 1)

	word_ngrams_2 = create_word_ngrams(preprocessed_corpus, 2)

	word_ngrams_3 = create_word_ngrams(preprocessed_corpus, 3)

	# compute distance coefficients for these ngrams
	wn_jaccard_1 = compute_jaccard(word_ngrams_1)

	wn_jaccard_2 = compute_jaccard(word_ngrams_2)

	wn_jaccard_3 = compute_jaccard(word_ngrams_3)


	wn_dice_1 = compute_dice(word_ngrams_1)

	wn_dice_2 = compute_dice(word_ngrams_2)

	wn_dice_3 = compute_dice(word_ngrams_3)


	wn_overlap_1 = compute_overlap(word_ngrams_1)

	wn_overlap_2 = compute_overlap(word_ngrams_2)

	wn_overlap_3 = compute_overlap(word_ngrams_3)

	# create character ngrams of different sizes
	character_ngrams_2 = create_character_ngrams(preprocessed_corpus, 2)

	character_ngrams_3 = create_character_ngrams(preprocessed_corpus, 3)

	character_ngrams_4 = create_character_ngrams(preprocessed_corpus, 4)

	# compute distance coefficients for these ngrams
	cn_jaccard_2 = compute_jaccard(character_ngrams_2)

	cn_jaccard_3 = compute_jaccard(character_ngrams_3)

	cn_jaccard_4 = compute_jaccard(character_ngrams_4)


	cn_dice_2 = compute_dice(character_ngrams_2)

	cn_dice_3 = compute_dice(character_ngrams_3)

	cn_dice_4 = compute_dice(character_ngrams_4)


	cn_overlap_2 = compute_overlap(character_ngrams_2)

	cn_overlap_3 = compute_overlap(character_ngrams_3)

	cn_overlap_4 = compute_overlap(character_ngrams_4)

	if word2vec_mdl:
		word2vec = w2c.word2vec_model(word2vec_mdl, corpus, 0, 1, 0)

		word2vec_tfidf = w2c.word2vec_model(word2vec_mdl, corpus, 1, 1, 0)

	if fasttext_mdl:
		fasttext = ftt.fasttext_model(fasttext_mdl, corpus, 0, 1, 0)

		fasttext_tfidf = ftt.fasttext_model(fasttext_mdl, corpus, 1, 1, 0)

	if ptlkb64_mdl:
		ptlkb_64 = ptlkb.ptlkb_model(ptlkb64_mdl, 0, 1, 0, lemmas)

		ptlkb_64_tfidf = ptlkb.ptlkb_model(ptlkb64_mdl, 1, 1, 0, lemmas)

	if glove300_mdl:
		glove_300 = ld_emb.word_embeddings_model(glove300_mdl, corpus, 0, 1, 0)

		glove_300_tfidf = ld_emb.word_embeddings_model(glove300_mdl, corpus, 1, 1, 0)

	# compute tfidf matrix - padding was applied to vectors of different sizes by adding zeros to the smaller vector of the pair
	tfidf_corpus = tl.preprocessing(corpus, 0, 0, 0, 1)
	tfidf_matrix = tl.compute_tfidf_matrix(tfidf_corpus, 0, 0, 1)

	# compute semantic relations coefficients
	semantic_relations = compute_semantic_relations(lemmas)

	# compute POS tags
	pos_tags = compute_pos(tags)

	# compute NERs
	ners = compute_ner(entities)

	# compute Syntactic Dependency parsing
	dependencies = dependency_parsing(corpus)

	# create multiple word ngrams of different sizes
	word_ngrams_1_2 = create_multiple_word_ngrams(preprocessed_corpus, 1, 2)

	# compute the cosine similarity between the multiple word ngrams converted sentences
	wn_cosine_1_2 = NG(word_ngrams_1_2)

	# create multiple character ngrams of different sizes
	character_ngrams_1_2_3 = create_multiple_character_ngrams(preprocessed_corpus, 1, 2, 3)

	# compute the cosine similarity between the multiple character ngrams converted sentences
	cn_cosine_1_2_3 = NG(character_ngrams_1_2_3)

	if numberbatch_mdl:
		numberbatch = ld_emb.word_embeddings_model(numberbatch_mdl, corpus, 0, 1, 0)

		numberbatch_tfidf = ld_emb.word_embeddings_model(numberbatch_mdl, corpus, 1, 1, 0)

	for pair in range(len(preprocessed_corpus)):
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
		if flag == 1:
			used_features[18] = True
			features_pair.append(word2vec[pair])
		if flag == 1:
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
		if flag == 1:
			used_features[26] = True
			features_pair.append(tfidf_matrix[pair])
		if flag == 1:
			used_features[27] = True
			features_pair.append(semantic_relations['antonyms'][pair])
		if flag == 1:
			used_features[28] = True
			features_pair.append(semantic_relations['synonyms'][pair])
		if flag == 1:
			used_features[29] = True
			features_pair.append(semantic_relations['hyperonyms'][pair])
		if flag == 1:
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
		if flag == 1 and ('n-adj' in pos_tags.columns):
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
		if flag == 1 and ('intj' in pos_tags.columns):
			used_features[46] = True
			features_pair.append(pos_tags['intj'][pair])
		else:
			features_pair.append(0)
		if flag == 1 and ('conj-s' in pos_tags.columns):
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
		if flag == 1 and ('all_ners' in ners.columns):
			used_features[50] = True
			features_pair.append(ners['all_ners'][pair])
		else:
			features_pair.append(0)
		if flag == 1 and ('B-ABSTRACCAO' in ners.columns):
			used_features[51] = True
			features_pair.append(ners['B-ABSTRACCAO'][pair])
		else:
			features_pair.append(0)
		if flag == 1 and ('B-ACONTECIMENTO' in ners.columns):
			used_features[52] = True
			features_pair.append(ners['B-ACONTECIMENTO'][pair])
		else:
			features_pair.append(0)
		if flag == 1 and ('B-COISA' in ners.columns):
			used_features[53] = True
			features_pair.append(ners['B-COISA'][pair])
		else:
			features_pair.append(0)
		if flag == 1 and ('B-LOCAL' in ners.columns):
			used_features[54] = True
			features_pair.append(ners['B-LOCAL'][pair])
		else:
			features_pair.append(0)
		if flag == 1 and ('B-OBRA' in ners.columns):
			used_features[55] = True
			features_pair.append(ners['B-OBRA'][pair])
		else:
			features_pair.append(0)
		if flag == 1 and ('B-ORGANIZACAO' in ners.columns):
			used_features[56] = True
			features_pair.append(ners['B-ORGANIZACAO'][pair])
		else:
			features_pair.append(0)
		if flag == 1 and ('B-OUTRO' in ners.columns):
			used_features[57] = True
			features_pair.append(ners['B-OUTRO'][pair])
		else:
			features_pair.append(0)
		if flag == 1 and ('B-PESSOA' in ners.columns):
			used_features[58] = True
			features_pair.append(ners['B-PESSOA'][pair])
		else:
			features_pair.append(0)
		if flag == 1 and ('B-TEMPO' in ners.columns):
			used_features[59] = True
			features_pair.append(ners['B-TEMPO'][pair])
		else:
			features_pair.append(0)
		if flag == 1 and ('B-VALOR' in ners.columns):
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
		if flag == 1:
			used_features[64] = True
			features_pair.append(numberbatch[pair])
		if flag == 1:
			used_features[65] = True
			features_pair.append(numberbatch_tfidf[pair])

		tuple_features_pair = tuple(features_pair)
		features.append(tuple_features_pair)

	return np.array(features), used_features

def load_features(f_selection, corpus, preprocessed_corpus, word2vec_mdl=None, fasttext_mdl=None, ptlkb64_mdl=None, glove300_mdl=None, numberbatch_mdl=None):
	""" Function used to extract the features """

	# start_time = time.time()
	# print("Started running the pipeline")

	pipeline_output = new_full_pipe(corpus, options={"string_or_array":True}, config_list=config_list)

	# there's still a bug in the pipeline that makes it output two EOS tokens at the end of each run, reason why we read the output to the penultimate token.
	tags = tl.build_sentences_from_tokens(pipeline_output.pos_tags)
	lemmas = tl.build_sentences_from_tokens(pipeline_output.lemas)
	entities = tl.build_sentences_from_tokens(pipeline_output.entities)

	# print("Finished running the pipeline successfully")
	# print("--- %s seconds ---" %(time.time() - start_time))
	# print('\a')

	features = []

	# create word ngrams of different sizes
	if 1 in f_selection[0:9:3]:
		word_ngrams_1 = create_word_ngrams(preprocessed_corpus, 1)

	if 1 in f_selection[1:9:3]:
		word_ngrams_2 = create_word_ngrams(preprocessed_corpus, 2)

	if 1 in f_selection[2:9:3]:
		word_ngrams_3 = create_word_ngrams(preprocessed_corpus, 3)

	# compute distance coefficients for these ngrams
	if f_selection[0]:
		wn_jaccard_1 = compute_jaccard(word_ngrams_1)
	if f_selection[1]:
		wn_jaccard_2 = compute_jaccard(word_ngrams_2)
	if f_selection[2]:
		wn_jaccard_3 = compute_jaccard(word_ngrams_3)

	if f_selection[3]:
		wn_dice_1 = compute_dice(word_ngrams_1)
	if f_selection[4]:
		wn_dice_2 = compute_dice(word_ngrams_2)
	if f_selection[5]:
		wn_dice_3 = compute_dice(word_ngrams_3)

	if f_selection[6]:
		wn_overlap_1 = compute_overlap(word_ngrams_1)
	if f_selection[7]:
		wn_overlap_2 = compute_overlap(word_ngrams_2)
	if f_selection[8]:
		wn_overlap_3 = compute_overlap(word_ngrams_3)

	# create character ngrams of different sizes
	if 1 in f_selection[9:18:3]:
		character_ngrams_2 = create_character_ngrams(preprocessed_corpus, 2)
	if 1 in f_selection[10:18:3]:
		character_ngrams_3 = create_character_ngrams(preprocessed_corpus, 3)
	if 1 in f_selection[11:18:3]:
		character_ngrams_4 = create_character_ngrams(preprocessed_corpus, 4)

	# compute distance coefficients for these ngrams
	if f_selection[9]:
		cn_jaccard_2 = compute_jaccard(character_ngrams_2)
	if f_selection[10]:
		cn_jaccard_3 = compute_jaccard(character_ngrams_3)
	if f_selection[11]:
		cn_jaccard_4 = compute_jaccard(character_ngrams_4)

	if f_selection[12]:
		cn_dice_2 = compute_dice(character_ngrams_2)
	if f_selection[13]:
		cn_dice_3 = compute_dice(character_ngrams_3)
	if f_selection[14]:
		cn_dice_4 = compute_dice(character_ngrams_4)

	if f_selection[15]:
		cn_overlap_2 = compute_overlap(character_ngrams_2)
	if f_selection[16]:
		cn_overlap_3 = compute_overlap(character_ngrams_3)
	if f_selection[17]:
		cn_overlap_4 = compute_overlap(character_ngrams_4)

	if word2vec_mdl:
		if f_selection[18]:
			word2vec = w2c.word2vec_model(word2vec_mdl, corpus, 0, 1, 0)
		if f_selection[19]:
			word2vec_tfidf = w2c.word2vec_model(word2vec_mdl, corpus, 1, 1, 0)

	if fasttext_mdl:
		if f_selection[20]:
			fasttext = ftt.fasttext_model(fasttext_mdl, corpus, 0, 1, 0)
		if f_selection[21]:
			fasttext_tfidf = ftt.fasttext_model(fasttext_mdl, corpus, 1, 1, 0)

	if ptlkb64_mdl:
		if f_selection[22]:
			ptlkb_64 = ptlkb.ptlkb_model(ptlkb64_mdl, 0, 1, 0, lemmas)
		if f_selection[23]:
			ptlkb_64_tfidf = ptlkb.ptlkb_model(ptlkb64_mdl, 1, 1, 0, lemmas)

	if glove300_mdl:
		if f_selection[24]:
			glove_300 = ld_emb.word_embeddings_model(glove300_mdl, corpus, 0, 1, 0)
		if f_selection[25]:
			glove_300_tfidf = ld_emb.word_embeddings_model(glove300_mdl, corpus, 1, 1, 0)

	# compute tfidf matrix - padding was applied to vectors of different sizes by adding zeros to the smaller vector of the pair
	if f_selection[26]:
		tfidf_corpus = tl.preprocessing(corpus, 0, 0, 0, 1)
		tfidf_matrix = tl.compute_tfidf_matrix(tfidf_corpus, 0, 0, 1)

	# compute semantic relations coefficients
	if 1 in f_selection[27:31]:
		semantic_relations = compute_semantic_relations(lemmas)

	# compute POS tags
	if 1 in f_selection[31:50]:
		pos_tags = compute_pos(tags)

	# compute NERs
	if 1 in f_selection[50:61]:
		ners = compute_ner(entities)

	# compute Syntactic Dependency parsing
	if f_selection[61]:
		dependencies = dependency_parsing(corpus)

	# create multiple word ngrams of different sizes
	if f_selection[62]:
		word_ngrams_1_2 = create_multiple_word_ngrams(preprocessed_corpus, 1, 2)

	# compute the cosine similarity between the multiple word ngrams converted sentences
	if f_selection[62]:
		wn_cosine_1_2 = NG(word_ngrams_1_2)

	# create multiple character ngrams of different sizes
	if f_selection[63]:
		character_ngrams_1_2_3 = create_multiple_character_ngrams(preprocessed_corpus, 1, 2, 3)

	# compute the cosine similarity between the multiple character ngrams converted sentences
	if f_selection[63]:
		cn_cosine_1_2_3 = NG(character_ngrams_1_2_3)

	if numberbatch_mdl:
		if f_selection[64]:
			numberbatch = ld_emb.word_embeddings_model(numberbatch_mdl, corpus, 0, 1, 0)
		if f_selection[65]:
			numberbatch_tfidf = ld_emb.word_embeddings_model(numberbatch_mdl, corpus, 1, 1, 0)

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

	return np.array(features)

def load_embeddings_models():
	""" Function used to load the word-embedding models """

	# ---LOADING WORD2VEC MODEL---
	model_load_path = os.path.join('models', 'word2vec', 'NILC', 'nilc_cbow_s300_300k.txt')
	start_time = time.time()
	print("Started loading the word2vec model")
	word2vec_model = KeyedVectors.load_word2vec_format(model_load_path)
	# word2vec_model = None
	print("Model loaded")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')

	# ---LOADING FASTTEXT MODEL---
	model_path = os.path.join('models', 'fastText', 'cc.pt.300_300k.vec')
	start_time = time.time()
	print("Started loading the fasttext model")
	fasttext_model = KeyedVectors.load_word2vec_format(model_path)
	# fasttext_model = None
	print("Model loaded")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')	

	# ---LOADING PT-LKB MODEL---
	model_load_path = os.path.join('models', 'ontoPT', 'PT-LKB_embeddings_64', 'ptlkb_64_30_200_p_str.emb')
	# model_load_path = os.path.join('models', 'ontoPT', 'PT-LKB_embeddings_128', 'ptlkb_128_80_10_p_str.emb')
	start_time = time.time()
	print("Started loading the PT-LKB-64 model")
	ptlkb64_model = KeyedVectors.load_word2vec_format(model_load_path)
	# ptlkb64_model = None
	print("Model loaded")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')

	# ---LOADING GLOVE-300 MODEL---
	model_load_path = os.path.join('models', 'glove', 'glove_s300_300k.txt')
	start_time = time.time()
	print("Started loading the GLOVE 300 dimensions model")
	glove300_model = KeyedVectors.load_word2vec_format(model_load_path)
	# glove300_model = None
	print("Model loaded")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')

	# ---LOADING NUMBERBATCH MODEL---
	model_load_path = os.path.join('models', 'numberbatch', 'numberbatch-17.02_pt_tratado.txt')
	start_time = time.time()
	print("Started loading the NUMBERBATCH dimensions model")
	numberbatch_model = KeyedVectors.load_word2vec_format(model_load_path)
	# numberbatch_model = None
	print("Model loaded")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')

	return word2vec_model, fasttext_model, ptlkb64_model, glove300_model, numberbatch_model

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
	"""

	system_mode = 4

	# Flag to indicate if the extracted features should be written to a file (1) or not (0)
	features_to_file_flag = 0

	# extract labels
	train_pairs = []
	# train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptpt-train.xml", need_labels=True))
	# train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptbr-train.xml", need_labels=True))

	if system_mode == 2 or system_mode == 5:
		train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptpt-test.xml", need_labels=True))
		train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptbr-test.xml", need_labels=True))
	if system_mode == 4 or system_mode == 5:
		train_pairs.extend(read_xml("datasets/assin/assin2/assin2-train-only.xml", need_labels=True))

	train_similarity_target = np.array([pair.similarity for pair in train_pairs])

	# extract training features
	train_corpus = tl.read_corpus(train_pairs)

	# tl.write_data_to_file(train_corpus, "finetune.train.raw")
	# print("Wrote training corpus")

	# preprocessing(text, tokenization=0, rm_stopwords=0, numbers_to_text=0, to_tfidf=0)
	preprocessed_train_corpus = tl.preprocessing(train_corpus, 0, 0, 0, 0)
	train_features, used_train_features = extract_features(train_corpus, preprocessed_train_corpus, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb64_mdl=ptlkb64_model, glove300_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

	# write train features to a .csv file
	if features_to_file_flag == 1:
		tl.write_features_to_csv(train_pairs, train_features, "assin1-train-test-assin2-train-ftrain.csv")

	#############################################################
	test_pairs_dev = read_xml('datasets/assin/assin2/assin2-dev.xml', need_labels=False)

	test_corpus_dev = tl.read_corpus(test_pairs_dev)
	# preprocessing(text, tokenization=0, rm_stopwords=0, numbers_to_text=0, to_tfidf=0)
	preprocessed_test_corpus_dev = tl.preprocessing(test_corpus_dev, 0, 0, 0, 0)
	test_features_dev, used_test_features_dev = extract_features(test_corpus_dev, preprocessed_test_corpus_dev, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb64_mdl=ptlkb64_model, glove300_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

	test_pairs_selection = read_xml('datasets/assin/assin2/assin2-dev.xml', need_labels=True)
	test_similarity_target = np.array([pair.similarity for pair in test_pairs_selection])
	#############################################################

	# extract test features
	# test_pairs = read_xml(args.test, need_labels=False)

	# uncomment next line and comment previous one to compute ASSIN 2 submission results
	test_pairs = read_xml_no_attributes(args.test)

	test_corpus = tl.read_corpus(test_pairs)
	# preprocessing(text, tokenization=0, rm_stopwords=0, numbers_to_text=0, to_tfidf=0)
	preprocessed_test_corpus = tl.preprocessing(test_corpus, 0, 0, 0, 0)
	test_features, used_test_features = extract_features(test_corpus, preprocessed_test_corpus, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb64_mdl=ptlkb64_model, glove300_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

	# write test features to a .csv file
	if features_to_file_flag == 1:
		tl.write_features_to_csv(test_pairs, test_features, "assin1-train-test-assin2-train-ftest.csv")

	# extract test features for feature selection (labels needed in order to perform evaluation)
	# test_pairs_selection = read_xml(args.test, need_labels=True)
	# test_similarity_target = np.array([pair.similarity for pair in test_pairs_selection])

	'''
	In order to create a STS model, it is required to select a regressor algorithm. We rely on the scikit-learn implementation of these algorithms to run our system. The default regresso is SVR, however an algorithm of your choice can be used. Select one type of regressor from scikit-learn. Here is a list with some examples: 
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

	use_feature_selection = 1

	if use_feature_selection:
		selected_selector, selected_train_features, selected_test_features = feature_selection(train_features, test_features_dev, train_similarity_target, test_similarity_target, regressor, used_train_features)

		test_features_selected = selected_selector.transform(test_features)

		regressor.fit(selected_train_features, train_similarity_target)

		# save model to disk
		model_save_path = os.path.join('trained_models', 'SVR_FS.joblib')
		dump(regressor, model_save_path)

		# apply model to the test dataset
		## this needs to be fixed in order to take advantage of the manual feature selection
		predicted_similarity = regressor.predict(test_features_selected)
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

# 	tl.write_data_to_file(train_corpus, "finetune.train.raw")
# 	print("Wrote training corpus")

	# test_pairs = read_xml_no_attributes("datasets/assin/assin2/assin2-blind-test.xml")

	# test_corpus = tl.read_corpus(test_pairs)

	# tl.write_data_to_file(test_corpus, "finetune.test.raw")
	# print("Wrote testing corpus")
