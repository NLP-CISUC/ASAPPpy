import os
import pandas as pd
import copy

from sklearn.metrics.pairwise import cosine_similarity
from gensim.parsing.preprocessing import strip_non_alphanum, strip_multiple_whitespaces
from random import randint

def read_faqs_variants():
	""" Function used to read the faqs variants """

	faqs_variants_load_path = os.path.join('datasets', 'AIA-BDE_v2.0.txt')

	with open(faqs_variants_load_path) as faqs_file:
		faqs_variants_corpus = faqs_file.read().splitlines()

	faqs_variants_corpus = [line.replace('\t', '') for line in faqs_variants_corpus]

	faqs_file.close()

	faqs_variants_corpus = [line.split(':', 1) for line in faqs_variants_corpus]
	faqs_variants_corpus = [line for line in faqs_variants_corpus if len(line) == 2 and line[1]]
	faqs_variants_corpus = [[line[0], strip_non_alphanum(line[1])] if line[0] != 'R' and line[0] != 'P' else [line[0], line[1]] for line in faqs_variants_corpus]
	faqs_variants_corpus = [[line[0].rstrip(), line[1].rstrip()] if line[0] != 'R' and line[0] != 'P' else [line[0], line[1]] for line in faqs_variants_corpus]
	faqs_variants_corpus = [[line[0], strip_multiple_whitespaces(line[1])] if line[0] != 'R' and line[0] != 'P' else [line[0], line[1]] for line in faqs_variants_corpus]
	faqs_variants_corpus = [[line[0], line[1].lower()] if line[0] != 'R' and line[0] != 'P' else [line[0], line[1]] for line in faqs_variants_corpus]

	OG = []
	VIN = []
	VG1 = []
	VG2 = []
	VUC = []
	VMT = []

	position = 0

	for element in faqs_variants_corpus:
		if element[0] == 'P':
			OG.append([element[1]])
			# the same element is added twice to the list in order to simulate the original question as a variant of its own
			OG[position].extend([element[1]])
			VIN.append([element[1]])
			VG1.append([element[1]])
			VG2.append([element[1]])
			VUC.append([element[1]])
			VMT.append([element[1]])

		if element[0] == 'VIN':
			VIN[position].extend([element[1]])

		if element[0] == 'VG1':
			VG1[position].extend([element[1]])

		if element[0] == 'VG2':
			VG2[position].extend([element[1]])

		if element[0] == 'VUC':
			VUC[position].extend([element[1]])

		if element[0] == 'VMT':
			VMT[position].extend([element[1]])

		if element[0] == 'R':
			OG[position].extend([element[1]])
			VIN[position].extend([element[1]])
			VG1[position].extend([element[1]])
			VG2[position].extend([element[1]])
			VUC[position].extend([element[1]])
			VMT[position].extend([element[1]])

			position += 1

	OG = [element for element in OG if len(element) > 2]
	VIN = [element for element in VIN if len(element) > 2]
	VG1 = [element for element in VG1 if len(element) > 2]
	VG2 = [element for element in VG2 if len(element) > 2]
	VUC = [element for element in VUC if len(element) > 2]
	VMT = [element for element in VMT if len(element) > 2]

	# write cleaned corpus to file
	# final_corpus = []
	# for element in VG1:
	#     for i in range(1, len(element)-1):
	#         final_corpus.extend([element[0], element[i]])

	# for element in VG2:
	#     for i in range(1, len(element)-1):
	#         final_corpus.extend([element[0], element[i]])

	# for element in VIN:
	#     for i in range(1, len(element)-1):
	#         final_corpus.extend([element[0], element[i]])

	# for element in VUC:
	#     for i in range(1, len(element)-1):
	#         final_corpus.extend([element[0], element[i]])

	# for element in VMT:
	#     for i in range(1, len(element)-1):
	#         final_corpus.extend([element[0], element[i]])

	# with open('FAQ_todas_variantes_texto_clean.txt', 'w') as clean_file:
	#     for element in final_corpus:
	#         clean_file.write("%s\n" % element)

	# clean_file.close()

	return OG, VIN, VG1, VG2, VUC, VMT

def read_class_set():
	df = pd.read_csv("datasets/divididossopcomlegendas.txt",sep='§',header=0)

	classes = df.values.tolist()

	class_1 = []
	class_2 = []
	class_3 = []
	class_4 = []

	for sentence in classes:
		if sentence[1] == 1:
			tmp_s = sentence[0].split(":")
			class_1.append(tmp_s[1])
		if sentence[1] == 10:
			tmp_s = sentence[0].split(":")
			class_2.append(tmp_s[1])
		if sentence[1] == 11:
			tmp_s = sentence[0].split(":")
			class_3.append(tmp_s[1])
		# this is the class of subtitles
		if sentence[1] == 15:
			class_4.append(sentence[0])

	return class_1, class_2, class_3

def pre_selection(unselected_phrases, model, position_correct_match):
	""" Function used to perform pre-selection """
	embeddings_unselected_phrases = []
	similaritires = []
	selected_phrases = []
	l_unselected_phrases = copy.deepcopy(unselected_phrases)

	for pair in l_unselected_phrases:
		temp_question = [model[word] for word in pair[0] if word in model.vocab]
		temp_variant = [model[word] for word in pair[1] if word in model.vocab]
		mean_question = sum(temp_question)/len(temp_question)
		mean_variant = sum(temp_variant)/len(temp_variant)

		embeddings_unselected_phrases.append([mean_question, mean_variant])

	for i in range(0, len(embeddings_unselected_phrases)):
		similarity = cosine_similarity([embeddings_unselected_phrases[i][0]], [embeddings_unselected_phrases[i][1]])
		similaritires.append(similarity[0][0])

	# compute the index of the 30 questions with the higher similarity
	selected_indexes = sorted(range(len(similaritires)), key=lambda x: similaritires[x])[-30:]

	# the correct match is added at the begining of the list
	if position_correct_match not in selected_indexes:
		# selected_indexes.insert(0, position_correct_match)
		# selected_phrases.extend(l_unselected_phrases[position_correct_match])

		# for j in range(len(selected_indexes)):
		#     selected_phrases.extend(l_unselected_phrases[selected_indexes[j]])
		selected_phrases = None
		selected_indexes = None
	else:
		selected_phrases.extend(l_unselected_phrases[position_correct_match])

		for j in range(len(selected_indexes)):
			if selected_indexes[j] != position_correct_match:
				selected_phrases.extend(l_unselected_phrases[selected_indexes[j]])
			else:
				selected_indexes.insert(0, selected_indexes.pop(j))

	return selected_phrases, selected_indexes

def n_max_elements(list1, N):
	""" Function to compute the N highest numbers of a list """
	n_list1 = copy.deepcopy(list1)
	final_list = []

	for i in range(0, N):
		max1 = 0

		for j in range(len(n_list1)):
			if n_list1[j] > max1:
				max1 = n_list1[j]

		n_list1.remove(max1)
		final_list.append(max1)

	return final_list

def n_max_elements_indexes(list1, N):
	""" Function to compute the N highest numbers of a list """
	# Note: Not the most efficient way of doing it

	n_list1 = copy.deepcopy(list1)
	elem_pos_list = []
	indexes_list = []

	for k in range(len(n_list1)):
		elem_pos_list.append([n_list1[k], k])

	for i in range(0, N):
		max_element = -1
		max_position = -1
		max_tmp_j = -1

		for j in range(len(elem_pos_list)):
			if elem_pos_list[j][0] > max_element:
				max_element = elem_pos_list[j][0]
				max_position = elem_pos_list[j][1]
				max_tmp_j = j

		indexes_list.append(max_position)
		elem_pos_list.remove(elem_pos_list[max_tmp_j])

	indexes_list.sort()

	return indexes_list

def select_multiple_random_subtitles():
	subtle_path = os.path.join('dataset', 'SubtleCorpusPTEN', 'por', 'corpus0sDialogues_clean.txt')
	selected_interactions = []
	selected_positions = []

	with open(subtle_path) as suble_file:
		subtle_corpus = suble_file.read().splitlines()

	suble_file.close()

	for i in range(0, 379):
		position = randint(0, len(subtle_corpus))

		if position not in selected_positions:
			# if len(subtle_corpus[position]) >= 4:
			selected_interactions.append(subtle_corpus[position])
			selected_positions.append(position)

	return selected_interactions

def write_output_file(model_path, accuracies, classif_flag, fselect_flag, preselect_flag):

	if '/' in model_path:
		split_model_path = model_path.split('/')
		model_name = split_model_path[1].split('.')

		filename = '{}_classif_{}_fselect_{}_preselect_{}.txt'.format(model_name[0], classif_flag, fselect_flag, preselect_flag)
	else:
		filename = '{}_classif_{}_fselect_{}_preselect_{}.txt'.format(model_path, classif_flag, fselect_flag, preselect_flag)

	filename_path = 'computed_accuracies/{}'.format(filename)

	if os.path.exists(filename_path):
		tmp_filename = input("The designated file already exists. Please insert a new filename: ")

		filename_path = 'computed_accuracies/{}.txt'.format(tmp_filename)

		while os.path.exists(filename_path):
			tmp_filename = input("The designated file already exists. Please insert a new filename: ")

			filename_path = 'computed_accuracies/{}.txt'.format(tmp_filename)

		with open(filename_path, 'w') as accuracies_file:
			for item in accuracies:
				accuracies_file.write("%s\n" % item)

		accuracies_file.close()

	else:
		with open(filename_path, 'w') as accuracies_file:
			for item in accuracies:
				accuracies_file.write("%s\n" % item)

		accuracies_file.close()
