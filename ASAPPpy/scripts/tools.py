'''
Module with auxiliary functions.
'''

from .xml_reader import read_xml

from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.matutils import Sparse2Corpus
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd
from num2words import num2words
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

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

def write_data_to_file(data, filename):
	""" Function used to debug the corpus state during preprocessing """
	if isinstance(data, pd.DataFrame):
		data_to_print = data.values.tolist()
	else:
		data_to_print = data

	with open(filename, 'w') as f:
		for item in data_to_print:
			f.write("%s\n" % item)

def write_features_to_csv(pairs, features, filename):
	""" Function used to write the features dataframe to a .csv file. """
	ids = []

	for pair in pairs:
		ids.append(pair.id)

	features_dataframe = pd.DataFrame(features)
	features_dataframe.insert(0, column="ID", value=ids)
	features_dataframe.to_csv(filename, index=False)

def deprecated_read_corpus(filename):
	""" Function used to read the corpus for training (old version). """
	train_list = read_xml(filename, 1)

	corpus = []

	for pair in train_list:
		corpus += [strip_multiple_whitespaces(pair.t)]
		corpus += [strip_multiple_whitespaces(pair.h)]

	return corpus

def read_corpus(pairs):
	""" Function used to read the corpus for training. """

	corpus = []

	for pair in pairs:
		corpus += [strip_multiple_whitespaces(pair.t)]
		corpus += [strip_multiple_whitespaces(pair.h)]

	return corpus

#pre-process the data
def preprocessing(text, tokenization=0, rm_stopwords=0, numbers_to_text=0, to_tfidf=0):
	""" Function used to preprocess the training data """
	train_data = pd.DataFrame(columns=['text', 'response'])

	prep_0 = [strip_non_alphanum(line) for line in text]
	prep_1 = [line for line in prep_0 if line.rstrip()]
	prep_2 = [strip_multiple_whitespaces(line) for line in prep_1]
	prep_3 = [line.lower() for line in prep_2]

	if to_tfidf == 1:
		#when using tf_idf, removes single character words given that they are ignored by sklearn's TfidfVectorizer
		prep_3 = [' '.join([word for word in line.split() if len(word) > 1]) for line in prep_3]

	if tokenization == 1:
		prep_3 = [line.split(' ') for line in prep_3]
		#removes whitespaces from the list
		prep_3 = [list(filter(None, line)) for line in prep_3]
	else:
		prep_3 = [line[:-1] if line[-1] == " " else line for line in prep_3]

	if numbers_to_text == 1 and tokenization == 1:
		#convert all numbers to integers and convert these numbers to its written form
		temp_prep = []
		for sentence in prep_3:
			temporary_sentence = []
			for word in sentence:
				if str(word).isdigit():
					converted_words = num2words(int(word), to='cardinal', lang='pt').split(' ')
					if to_tfidf == 1 and rm_stopwords == 0:
						converted_words = [word for word in converted_words if word != 'e']
					temporary_sentence.extend(converted_words)
				else:
					temporary_sentence.append(word)
			temp_prep.append(temporary_sentence)

		prep_3 = temp_prep
	elif numbers_to_text == 1 and tokenization == 0:
		#convert all numbers to integers and convert these numbers to its written form
		temp_prep = []
		for sentence in prep_3:
			temporary_sentence = []
			for word in sentence.split(' '):
				if str(word).isdigit():
					converted_words = num2words(int(word), to='cardinal', lang='pt').split(' ')
					if to_tfidf == 1 and rm_stopwords == 0:
						converted_words = [word for word in converted_words if word != 'e']
					temporary_sentence.extend(converted_words)
				else:
					temporary_sentence.append(word)
			temporary_sentence = ' '.join(temporary_sentence)
			temp_prep.append(temporary_sentence)
		prep_3 = temp_prep

	if rm_stopwords == 1:
		stp = set(stopwords.words('portuguese') + list(punctuation))
		if tokenization == 1:
			prep_3 = [[word for word in sentence if word not in stp] for sentence in prep_3]
		elif tokenization == 0:
			prep_3 = [' '.join([word for word in sentence.split(' ') if word not in stp]) for sentence in prep_3]

	tmp = pd.DataFrame({'text':prep_3[::2], 'response':prep_3[1::2]})
	train_data = train_data.append(tmp[['text', 'response']], ignore_index=True)

	return train_data

def compute_tfidf_matrix(text, remove_stopwords=0, to_gensim=0, compute_cos_similarity=0):
	""" Function used to compute the TF-IDF Matrix of the corpus """
	stp = set(stopwords.words('portuguese') + list(punctuation))

	list_1 = list(text['text'])
	list_2 = list(text['response'])

	corpus = list_1 + list_2
	corpus[::2] = list_1
	corpus[1::2] = list_2

	if remove_stopwords == 0:
		vectorizer = TfidfVectorizer(stop_words=None, min_df=0, max_features=None)
	else:
		vectorizer = TfidfVectorizer(stop_words=stp, min_df=0, max_features=None)

	tfidf_matrix = vectorizer.fit_transform(corpus)

	feature_names = vectorizer.get_feature_names()

	#word2tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

	#for word, score in word2tfidf.items():
		#print(word, score)

	if to_gensim:
		tfidf_matrix_list = list(Sparse2Corpus(tfidf_matrix, documents_columns=False))

		#convert gensim format to list
		temp_tfidf_matrix_list = []
		for sentence in tfidf_matrix_list:

			temp_sentence = []
			for word in sentence:
				temp_word = feature_names[word[0]]
				temp_pair = (temp_word, word[1])

				temp_sentence.append(temp_pair)
			temp_tfidf_matrix_list.append(dict(temp_sentence))

		tfidf_matrix_list = temp_tfidf_matrix_list

		return tfidf_matrix_list

	if compute_cos_similarity:
		tfidf_matrix_list = list(Sparse2Corpus(tfidf_matrix, documents_columns=False))

		#convert gensim format to list
		temp_tfidf_matrix_list = []
		for sentence in tfidf_matrix_list:

			temp_sentence = []
			for word in sentence:
				temp_pair = word[1]

				temp_sentence.append(temp_pair)
			temp_tfidf_matrix_list.append(temp_sentence)

		tfidf_matrix_list = temp_tfidf_matrix_list

		similarity_m = []

		for i in range(0, len(tfidf_matrix_list), 2):
			if len(tfidf_matrix_list[i]) > len(tfidf_matrix_list[i+1]):
				difference_in_length = len(tfidf_matrix_list[i]) - len(tfidf_matrix_list[i+1])
				tfidf_matrix_list[i+1].extend([0] * difference_in_length)
			elif len(tfidf_matrix_list[i]) < len(tfidf_matrix_list[i+1]):
				difference_in_length = len(tfidf_matrix_list[i+1]) - len(tfidf_matrix_list[i])
				tfidf_matrix_list[i].extend([0] * difference_in_length)

			similarity = cosine_similarity([tfidf_matrix_list[i]], [tfidf_matrix_list[i+1]])
			similarity_m.append(similarity[0][0])

		return similarity_m

	return tfidf_matrix

# if __name__ == "__main__":
# 	corpus = deprecated_read_corpus("assin-ptpt-test.xml")

# 	#function arguments
# 	remove_stopwords = 0
# 	convert_num_to_text = 0
# 	tf_idf = 1
# 	# when it comes to word2vec, the preprocessing function should always have tokenization equal to 1
# 	preprocessed_corpus = preprocessing(corpus, 0, remove_stopwords, convert_num_to_text, tf_idf)

# 	print(preprocessed_corpus)

# 	matrix = compute_tfidf_matrix(preprocessed_corpus, remove_stopwords, 0)
# 	tfidf_matrix_list = list(Sparse2Corpus(matrix, documents_columns=False))

# 	#convert gensim format to list
# 	temp_tfidf_matrix_list = []
# 	for sentence in tfidf_matrix_list:

# 		temp_sentence = []
# 		for word in sentence:
# 			temp_pair = word[1]

# 			temp_sentence.append(temp_pair)
# 		temp_tfidf_matrix_list.append(temp_sentence)

# 	tfidf_matrix_list = temp_tfidf_matrix_list

# 	similarity_m = []
# 	sized = int(len(tfidf_matrix_list)/2)

# 	for i in range(0, sized, 2):
# 		if len(tfidf_matrix_list[i]) > len(tfidf_matrix_list[i+1]):
# 			difference_in_length = len(tfidf_matrix_list[i]) - len(tfidf_matrix_list[i+1])
# 			tfidf_matrix_list[i+1].extend([0] * difference_in_length)
# 		elif len(tfidf_matrix_list[i]) < len(tfidf_matrix_list[i+1]):
# 			difference_in_length = len(tfidf_matrix_list[i+1]) - len(tfidf_matrix_list[i])
# 			tfidf_matrix_list[i].extend([0] * difference_in_length)

# 		similarity = cosine_similarity([tfidf_matrix_list[i]], [tfidf_matrix_list[i+1]])
# 		similarity_m.append(similarity[0][0])
