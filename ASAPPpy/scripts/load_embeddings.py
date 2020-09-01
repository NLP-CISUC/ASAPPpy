import os
import time
from gensim.models import KeyedVectors

from ..__init__ import ROOT_PATH

def load_embeddings_models():
	""" Function used to load the word-embedding models """

	# ---LOADING WORD2VEC MODEL---
	model_load_path = os.path.join(ROOT_PATH, 'models', 'word2vec', 'NILC', 'nilc_cbow_s300_300k.txt')
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
	start_time = time.time()
	print("Started loading the PT-LKB-64 model")
	ptlkb64_model = KeyedVectors.load_word2vec_format(model_load_path)
	# ptlkb64_model = None
	print("Model loaded")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')

	# ---LOADING GLOVE-300 MODEL---
	model_load_path = os.path.join(ROOT_PATH, 'models', 'glove', 'glove_s300_300k.txt')
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
