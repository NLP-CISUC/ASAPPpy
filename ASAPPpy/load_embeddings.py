'''
Module used to compute similarity between sentences of the ASSIN collection using word embeddings.
'''

import time
import os
from xml.etree import cElementTree as ET
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

#import function to read the xml file
from ASAPPpy.scripts.xml_reader import read_xml
from ASAPPpy.scripts.tools import preprocessing, compute_tfidf_matrix

global n_iterations
n_iterations = 0

def apply_tfidf_model(sentence, embedding_model, tfidf_model):
	""" Auxiliar function used to compute the vector of each sentence by applying the word2vec model and the tfidf model """

	global n_iterations

	updated_sentence = []

	for word in sentence:
		if word in embedding_model.vocab:
			new_word = embedding_model[word] * tfidf_model[n_iterations][word]
			updated_sentence.append(new_word)

	n_iterations += 2
	return updated_sentence

def compute_models(model, embeddings_data, tfidf_data, use_tf_idf=0, rm_stopwords=0, numbers_to_text=0):
	""" Function used to apply the word embeddings model and compute the vector of each sentence """

	global n_iterations

	if use_tf_idf == 1:

		# tokenization should be 0 when applying preprocessing to the data that will be used to build the tfidf matrix
		tfidf_preprocessed = preprocessing(tfidf_data, 0, rm_stopwords, numbers_to_text, use_tf_idf)
		tfidf_model = compute_tfidf_matrix(tfidf_preprocessed, rm_stopwords, 1, 0)

		n_iterations = 0
		embeddings_data['text'] = embeddings_data['text'].apply(lambda x: apply_tfidf_model(x, model, tfidf_model))
		n_iterations = 1
		embeddings_data['response'] = embeddings_data['response'].apply(lambda x: apply_tfidf_model(x, model, tfidf_model))
		n_iterations = 0
	else:
		# The NILC embeddings don't allow to increment the existing model with new data, because it
		# doesn't have a binary version of it.

		embeddings_data['text'] = embeddings_data['text'].apply(lambda x: [model[word] for word in x if word in model.vocab])
		embeddings_data['response'] = embeddings_data['response'].apply(lambda x: [model[word] for word in x if word in model.vocab])

	#compute the mean of the sentence
	embeddings_data['text'] = embeddings_data['text'].apply(lambda x: sum(x)/len(x) if len(x) != 0 else [0]*300)
	embeddings_data['response'] = embeddings_data['response'].apply(lambda x: sum(x)/len(x) if len(x) != 0 else [0]*300)

	return embeddings_data

def word_embeddings_model(model, corpus, tf_idf, remove_stopwords, convert_num_to_text):
	""" Function used to load the word embeddings model and compute the vector of each sentence """

	word_embeddings = []

	# when it comes to word embeddings, the preprocessing function should always have tokenization equal to 1
	preprocessed_corpus = preprocessing(corpus, 1, remove_stopwords, convert_num_to_text, tf_idf)

	if tf_idf == 1:
		embeddings_data = compute_models(model, preprocessed_corpus, corpus, tf_idf, remove_stopwords, convert_num_to_text)
	else:
		embeddings_data = compute_models(model, preprocessed_corpus, corpus, tf_idf)

	for i in range(len(embeddings_data)):
		similarity = cosine_similarity([embeddings_data['text'][i]], [embeddings_data['response'][i]])
		word_embeddings.append(similarity[0][0])

	return word_embeddings
