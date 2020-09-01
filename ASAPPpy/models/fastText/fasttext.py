'''
Module to compute the fasttext similarity in the ASSIN collection.
'''

import time
import os
from xml.etree import cElementTree as ET
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

from ...scripts.tools import preprocessing, compute_tfidf_matrix, deprecated_read_corpus

global n_iterations
n_iterations = 0

def apply_tfidf_model(sentence, embedding_model, tfidf_model):
	""" Auxiliar function used to compute the vector of each sentence by applying the word2vec model and the tfidf model"""

	global n_iterations

	updated_sentence = []

	for word in sentence:
		if word in embedding_model.vocab:
			new_word = embedding_model[word] * tfidf_model[n_iterations][word]
			updated_sentence.append(new_word)

	n_iterations += 2
	return updated_sentence

def compute_models(model, embeddings_data, tfidf_data, use_tf_idf=0, rm_stopwords=0, numbers_to_text=0):
	""" Function used to load the FastText model and compute the vector of each sentence """

	global n_iterations

	if use_tf_idf == 1:

		# tokenization should be 0 when applying preprocessing to the date that will be used to build the tfidf matrix
		tfidf_preprocessed = preprocessing(tfidf_data, 0, rm_stopwords, numbers_to_text, use_tf_idf)
		tfidf_model = compute_tfidf_matrix(tfidf_preprocessed, rm_stopwords, 1, 0)

		n_iterations = 0
		embeddings_data['text'] = embeddings_data['text'].apply(lambda x: apply_tfidf_model(x, model, tfidf_model))
		n_iterations = 1
		embeddings_data['response'] = embeddings_data['response'].apply(lambda x: apply_tfidf_model(x, model, tfidf_model))
		n_iterations = 0
	else:
		#convert words to vectors
		embeddings_data['text'] = embeddings_data['text'].apply(lambda x: [model[word] for word in x if word in model.vocab])
		embeddings_data['response'] = embeddings_data['response'].apply(lambda x: [model[word] for word in x if word in model.vocab])

	#compute the mean of the sentence
	embeddings_data['text'] = embeddings_data['text'].apply(lambda x: sum(x)/len(x) if len(x) != 0 else [0]*300)
	embeddings_data['response'] = embeddings_data['response'].apply(lambda x: sum(x)/len(x) if len(x) != 0 else [0]*300)

	return embeddings_data

def fasttext_model(model, corpus, tf_idf, remove_stopwords, convert_num_to_text):
	fasttext_embeddings = []

	# when it comes to fasttext, the preprocessing function should always have tokenization equal to 1
	preprocessed_corpus = preprocessing(corpus, 1, remove_stopwords, convert_num_to_text, tf_idf)

	if tf_idf == 1:
		embeddings_data = compute_models(model, preprocessed_corpus, corpus, tf_idf, remove_stopwords, convert_num_to_text)
	else:
		embeddings_data = compute_models(model, preprocessed_corpus, corpus, tf_idf)

	for i in range(len(embeddings_data)):
		similarity = cosine_similarity([embeddings_data['text'][i]], [embeddings_data['response'][i]])
		fasttext_embeddings.append(similarity[0][0])

	return fasttext_embeddings

def fasttext_model_single_feature():
	""" Function used to compute similarity """

	# ---LOADING FASTTEXT MODEL---
	model_path = os.path.join('models', 'fastText', 'cc.pt.300.vec')
	start_time = time.time()
	#model = FastText.load_fasttext_format(model_path)
	#model.build_vocab(train_data, update=True)
	#model.train(train_data, epochs=model.iter, total_examples=model.corpus_count)
	print("Started loading the model")
	model = KeyedVectors.load_word2vec_format(model_path)
	print("Model loaded")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')

	write_path = os.path.join('assin', 'assin_results', 'fasttext_facebook_pretrained_stp_rmv_num_conv_tfidf.xml')

	test_list = read_xml("assin-ptpt-test.xml", 1)

	corpus = deprecated_read_corpus("assin-ptpt-test.xml")

	#function arguments
	remove_stopwords = 1
	convert_num_to_text = 1
	tf_idf = 1
	# when it comes to fasttext, the preprocessing function should always have tokenization equal to 1
	preprocessed_corpus = preprocessing(corpus, 1, remove_stopwords, convert_num_to_text, tf_idf)

	if tf_idf == 1:
		test_data = compute_models(model, preprocessed_corpus, corpus, tf_idf, remove_stopwords, convert_num_to_text)
	else:
		test_data = compute_models(model, preprocessed_corpus, corpus, tf_idf)

	for i in range(len(test_data)):
		similarity = cosine_similarity([test_data['text'][i]], [test_data['response'][i]])
		test_list[i].similarity = similarity[0][0]

	# write output
	tree = ET.parse("assin-ptpt-test.xml")
	root = tree.getroot()
	for i in range(len(test_list)):
		pairs = root[i]
		pairs.set('similarity', str(test_list[i].similarity))

	tree.write(write_path, 'utf-8')

#fasttext_model_single_feature()
