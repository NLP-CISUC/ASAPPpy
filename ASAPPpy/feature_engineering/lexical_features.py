'''
Module used to extract lexical features from a given corpus.
'''

import copy

from nltk import ngrams
from sklearn.metrics.pairwise import cosine_similarity

def aux_create_character_ngrams(sentence, number_of_grams):
	""" Auxiliar function used to create character ngrams, in order for them to be restricted to the boundaries of the word """
	allgrams = []
	for word in sentence.split():
		word_ngrams = list(ngrams(word, number_of_grams))

		for gram in word_ngrams:
			joint_gram = ''.join(gram)
			allgrams.append(joint_gram)
	return allgrams

def create_character_ngrams(original_text, number_of_grams):
	""" Function used to create character ngrams """
	text = copy.deepcopy(original_text)
	#during the preprocessing step, tokenization should not be applied

	#remove all words that are smaller than the least number of characters the n-gram will have
	text['text'] = text['text'].apply(lambda x: ' '.join([word for word in x.split() if len(word) >= number_of_grams]))
	text['response'] = text['response'].apply(lambda x: ' '.join([word for word in x.split() if len(word) >= number_of_grams]))

	#produces the character ngrams
	text['text'] = text['text'].apply(lambda x: aux_create_character_ngrams(x, number_of_grams))
	text['response'] = text['response'].apply(lambda x: aux_create_character_ngrams(x, number_of_grams))

	#Manual alternative, it could be useful coming back to it if something goes wrong with the current implementation
	#text['text'] = text['text'].apply(lambda x: [x[i:i+number_of_grams] for i in range(len(x)-number_of_grams+1)])
	#text['response'] = text['response'].apply(lambda x: [x[i:i+number_of_grams] for i in range(len(x)-number_of_grams+1)])

	return text

def aux_create_multiple_character_ngrams(sentence, number_of_grams):
	""" Auxiliar function used to create the union of multiple character ngrams of a sentence """
	all_grams = []

	for number in number_of_grams:
		temp_grams = []

		for word in sentence.split():
			if len(word) >= number:
				word_ngrams = list(ngrams(word, number))

				for gram in word_ngrams:
					joint_gram = ''.join(gram)
					temp_grams.append(joint_gram)
		all_grams.extend(temp_grams)

	return all_grams

def create_multiple_character_ngrams(original_text, *number_of_grams):
	""" Function used to create the union of multiple character ngrams of a sentence """
	text = copy.deepcopy(original_text)

	#produces the character ngrams
	text['text'] = text['text'].apply(lambda x: aux_create_multiple_character_ngrams(x, number_of_grams))
	text['response'] = text['response'].apply(lambda x: aux_create_multiple_character_ngrams(x, number_of_grams))

	return text

def create_word_ngrams(original_text, number_of_grams):
	""" Function used to create word ngrams """
	text = copy.deepcopy(original_text)
	#during the preprocessing step, tokenization should not be applied
	text['text'] = text['text'].apply(lambda x: list(ngrams(x.split(), number_of_grams)))
	text['response'] = text['response'].apply(lambda x: list(ngrams(x.split(), number_of_grams)))

	return text

def aux_create_multiple_word_ngrams(sentence, number_of_grams):
	""" Auxiliar function used to create the union of multiple word ngrams of a sentence """
	all_grams = []
	string_all_grams = []
	splitted_sentence = sentence.split()

	for number in number_of_grams:
		all_grams.extend(list(ngrams(splitted_sentence, number)))

	for gram in all_grams:
		temp_string = ' '.join(gram)
		string_all_grams.append(temp_string)

	return string_all_grams

def create_multiple_word_ngrams(original_text, *number_of_grams):
	""" Function used to create the union of multiple word ngrams of a sentence """
	text = copy.deepcopy(original_text)
	#during the preprocessing step, tokenization should not be applied
	text['text'] = text['text'].apply(lambda x: aux_create_multiple_word_ngrams(x, number_of_grams))
	text['response'] = text['response'].apply(lambda x: aux_create_multiple_word_ngrams(x, number_of_grams))

	return text

def jaccard_coefficient(label1, label2):
	""" Distance metric comparing set-similarity. """
	#the original formula used float, but it doesn't seem to make any difference
	#return float(len(label1 & label2)) / min(len(label1), len(label2))
	if len(label1.union(label2)) == 0:
		return 0
	else:
		return len(label1.intersection(label2)) / len(label1.union(label2))

def compute_jaccard(text):
	""" Function used to compute the Jaccard distance between two sentences in the corpus """
	#print(text)
	#during the preprocessing step, tokenization should be applied
	text['jaccard'] = text.apply(lambda pair: jaccard_coefficient(set(pair['text']), set(pair['response'])), axis=1)
	return text

def overlap_coefficient(label1, label2):
	""" Distance metric comparing set-similarity. """
	#the original formula used float, but it doesn't seem to make any difference
	#return float(len(label1 & label2)) / min(len(label1), len(label2))
	if min(len(label1), len(label2)) == 0:
		return 0
	else:
		return len(label1.intersection(label2)) / min(len(label1), len(label2))

def compute_overlap(text):
	#print(text)
	""" Function used to compute the Overlap coefficient between two sentences in the corpus """
	text['overlap'] = text.apply(lambda pair: overlap_coefficient(set(pair['text']), set(pair['response'])), axis=1)
	return text

def dice_distance(label1, label2):
	""" Distance metric comparing set-similarity. """
	if (len(label1) + len(label2)) == 0:
		return 0
	else:
		return len(label1.intersection(label2)) / (len(label1) + len(label2))
    #return (len(label1.union(label2)) - len(label1.intersection(label2))) / len(label1.union(label2))

def compute_dice(text):
	""" Function used to compute the Dice coefficient between two sentences in the corpus """
	text['dice'] = text.apply(lambda pair: dice_distance(set(pair['text']), set(pair['response'])), axis=1)
	return text

def ng_cosine(label1, label2):
	""" Cosine similarity between pairs of sentences. """
	pair_gram_union = label1.union(label2)

	label1_vector = []
	label2_vector = []

	if len(pair_gram_union) == 0:
		return 0
	else:
		for gram in pair_gram_union:
			if gram in label1:
				label1_vector.append(1)
			else:
				label1_vector.append(0)

			if gram in label2:
				label2_vector.append(1)
			else:
				label2_vector.append(0)

		similarity = cosine_similarity([label1_vector], [label2_vector])

		return similarity[0][0]

def NG(text):
	""" Function used to compute the Cosine Similarity between two sentences in the corpus """
	text['NG'] = text.apply(lambda pair: ng_cosine(set(pair['text']), set(pair['response'])), axis=1)
	return text
