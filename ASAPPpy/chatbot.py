'''
The Chatbot Module
'''

import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from joblib import load
from sklearn import metrics

from resources.resources import read_faqs_variants, read_class_set, pre_selection, n_max_elements, n_max_elements_indexes, select_multiple_random_subtitles, write_output_file
# from feature_extraction import extract_features
from tmp_feature_extraction import extract_features, load_features
from classifiers.svm import svm_classifier
from classifiers.svm_restantes_classes import corre_para_testes_restantes
from indexers.Whoosh.whoosh_make_query import query_indexer
from ASAPPpy.sts_model import STSModel

def chatbot(word2vec_model=None, fasttext_model=None, ptlkb64_model=None, ptlkb128_model=None, glove100_model=None, glove300_model=None, numberbatch_model=None):
	"""
	Function used to run the chatbot

	"""

	# Flag to indicate if classification should be used (1) or not (0)
	classification_flag = 0

	# Choose if pre-selection should not be used (0), used with word embeddings (1) or used with Whoosh (2)
	pre_selection_flag = 2

	# Choose whether (1) or not (0) a new file with VUC random positions should be generated
	generate_random_positions_file_flag = 0

	# load the pre-trained STS model
	model_name = 'model_1906_ablation_study_master'

	model = STSModel()
	model.load_model(model_name)

	if classification_flag:
		print("The classifier is being used.")

		# read the different class sets
		class_1, class_2, class_3 = read_class_set()

		# transform the class sets from lists to dataframes
		class_1_df = pd.DataFrame(class_1, columns=['text'])
		class_2_df = pd.DataFrame(class_2, columns=['text'])
		class_3_df = pd.DataFrame(class_3, columns=['text'])

	OG, VIN, VG1, VG2, VUC, VMT = read_faqs_variants()

	# to test the system different sets of FAQs' variants (VIN, VG1, VG2, VUC, VMT) can be used, as well as the original questions (OG). To do so, they should be in the form: all_variants = [OG, VIN, VG1, VG2, VUC, VMT] and all_variants_names = ["OG", "VIN", "VG1", "VG2", "VUC", "VMT"]
	all_variants = [OG, VIN, VG1, VG2, VUC, VMT]
	all_variants_names = ["OG", "VIN", "VG1", "VG2", "VUC", "VMT"]

	all_accuracies = [['Variant collection', 'Mean', 'Number of Correct matches', 'Total number of questions', "Top 3", "Top 5", "Threshold 0", "Threshold 1", "Threshold 1,5", "Threshold 2", "Threshold 2,5", "Threshold 3", "Threshold 3,5", "Threshold 4",]]

	if generate_random_positions_file_flag == 1:
		random_positions_VUC = []

	incorrect_matches_VG1 = []
	use_only_one_vuc = 0

	if use_only_one_vuc:
		with open('random_positions_VUC.txt') as random_positions_VUC_file:
			random_positions_VUC = random_positions_VUC_file.read().splitlines()

		random_positions_VUC_file.close()

	# given that que questions indexed with Whoosh were preprocessed, the original questions are needed in order to confirm if the correct response was returned
	if pre_selection_flag == 2:
		original_questions_with_duplicates = [phrases[0] for phrases in OG]

		original_questions = []
		for question in original_questions_with_duplicates:
			if question not in original_questions:
				original_questions.append(question)

	# used to plot the ROC curve
	y_true = []
	y_score = []

	for pos_variant, variant in enumerate(all_variants):

		if all_variants_names[pos_variant] == 'VIN':
			start_time = time.time()
			print("Started testing the chatbot")

		list_of_questions = [phrases[0] for phrases in variant]
		questions_df = pd.DataFrame(list_of_questions, columns=['text'])

		n_correct_matches = 0
		n_variant_questions = 0
		n_top3 = 0
		n_top5 = 0
		threshold_0 = 0
		threshold_1 = 0
		threshold_1_5 = 0
		threshold_2 = 0
		threshold_2_5 = 0
		threshold_3 = 0
		threshold_3_5 = 0
		threshold_4 = 0

		for position, element in enumerate(variant):
			print('Variant %s - %d/%d' % (all_variants_names[pos_variant], position+1, len(all_variants[pos_variant])), end='\r')

			for i in range(1, len(element)-1):
				unprocessed_corpus = []
				n_variant_questions += 1

				# select a random variant from VUC
				if all_variants_names[pos_variant] == 'VUC' and use_only_one_vuc:
					i = int(random_positions_VUC[position])

					if generate_random_positions_file_flag == 1:
						i = randint(1, len(element)-2)
						random_positions_VUC.append(i)
						i = 1

				if classification_flag:
					# apply the classifier before using the STS model
					# predicted_class = svm_classifier(element[i])
					predicted_class = corre_para_testes_restantes([element[i]])

					if predicted_class == 1:
						aux_df = class_1_df
						aux_list_of_questions = class_1
					elif predicted_class == 2:
						aux_df = class_2_df
						aux_list_of_questions = class_2
					elif predicted_class == 3:
						aux_df = class_3_df
						aux_list_of_questions = class_3
					else:
						aux_df = questions_df
						aux_list_of_questions = list_of_questions
				else:
					aux_df = questions_df
					aux_list_of_questions = list_of_questions

				if 'response' not in aux_df:
					aux_df.insert(1, 'response', element[i])
				else:
					aux_df['response'] = element[i]

				if pre_selection_flag != 2:
					for j in range(len(aux_list_of_questions)):
						if int(pre_selection_flag) == 1:
							unprocessed_corpus.append([aux_list_of_questions[j], element[i]])
						else:
							unprocessed_corpus.extend([aux_list_of_questions[j], element[i]])

				if pre_selection_flag == 1:
					corpus_pairs, indexes = pre_selection(unprocessed_corpus, fasttext_model, position)

					if corpus_pairs is None:
						if all_variants_names[pos_variant] == 'VUC' and use_only_one_vuc:
							break
						continue

					selected_aux_df = aux_df.iloc[indexes]
					selected_aux_df = selected_aux_df.reset_index(drop=True)
				else:
					if pre_selection_flag == 2:
						index_path = os.path.join('indexers', 'Whoosh', 'indexes', 'FAQs_stemming_analyser_charset_filter_AIA-BDE_v2.0_no_preprocessing')

						query_response = query_indexer(element[i], index_path)
						options_docnumbers = query_response[2]

						if len(options_docnumbers) == 0:
							continue
						
						aux_corpus_pairs = []

						for pos, elem in enumerate(options_docnumbers):
							unprocessed_corpus.extend([original_questions[elem], element[i]])
							#aux_corpus_pairs.append([original_questions[elem], element[i]])
							aux_corpus_pairs.append(original_questions[elem])

					corpus_pairs = unprocessed_corpus
					selected_aux_df = aux_df

				element_features = model.extract_multiple_features(corpus_pairs, 0, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb_mdl=ptlkb64_model, glove_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

				# number_of_features_train = converted_mask.count(1)
				# number_of_features_test = len(element_features[0])

				# if number_of_features_train > number_of_features_test:
				# 	element_features_list = element_features.tolist()
				# 	for feature_pair in element_features_list:
				# 		feature_pair.extend([0]*(number_of_features_train-number_of_features_test))
				# 	element_features = np.asarray(element_features_list)

				predicted_similarity = model.predict_similarity(element_features)
				predicted_similarity = predicted_similarity.tolist()

				highest_match = max(predicted_similarity)
				highest_match_index = predicted_similarity.index(max(predicted_similarity))

				# if pre_selection is used the correct match will always be in the first position
				if int(pre_selection_flag) == 1:
					if highest_match_index == 0:
						n_correct_matches += 1
						if 0 <= highest_match < 1:
							threshold_0 += 1
						if 1 <= highest_match < 1.5:
							threshold_1 += 1
						if 1.5 <= highest_match < 2:
							threshold_1_5 += 1
						if 2 <= highest_match < 2.5:
							threshold_2 += 1
						if 2.5 <= highest_match < 3:
							threshold_2_5 += 1
						if 3 <= highest_match < 3.5:
							threshold_3 += 1
						if 3.5 <= highest_match < 4:
							threshold_3_5 += 1
						if 4 <= highest_match < 5:
							threshold_4 += 1
					else:
						top3 = n_max_elements(predicted_similarity, 3)
						top5 = n_max_elements(predicted_similarity, 5)

						if predicted_similarity[0] in top3:
							n_top3 += 1

						if (predicted_similarity[0] in top5) and (predicted_similarity[0] not in top3):
							n_top5 += 1

						# for research purposes only. Used to evaluate why mismatches happen with VG1
						if all_variants_names[pos_variant] == 'VG1':
							incorrect_matches_VG1.append([element[0], element[i], aux_list_of_questions[highest_match_index], highest_match])
				else:
					# if highest_match_index == position:
					if pre_selection_flag == 0:
						match = aux_list_of_questions[highest_match_index]
					elif pre_selection_flag == 2:
						match = aux_corpus_pairs[highest_match_index]
					else:
						print("Pre-selection flag was assigned a wrong number. Please fix it and re-run the script.")

					if match == element[0]:
						# used to plot the ROC curve
						y_true.append(1)
						y_score.append(highest_match)

						n_correct_matches += 1
						if 0 <= highest_match < 1:
							threshold_0 += 1
						if 1 <= highest_match < 1.5:
							threshold_1 += 1
						if 1.5 <= highest_match < 2:
							threshold_1_5 += 1
						if 2 <= highest_match < 2.5:
							threshold_2 += 1
						if 2.5 <= highest_match < 3:
							threshold_2_5 += 1
						if 3 <= highest_match < 3.5:
							threshold_3 += 1
						if 3.5 <= highest_match < 4:
							threshold_3_5 += 1
						if 4 <= highest_match < 5:
							threshold_4 += 1
					else:
						# used to plot the ROC curve
						y_true.append(0)
						y_score.append(highest_match)

						if pre_selection_flag == 0:
							top3 = n_max_elements(predicted_similarity, 3)
							top5 = n_max_elements(predicted_similarity, 5)

							if predicted_similarity[position] in top3:
								n_top3 += 1

							if (predicted_similarity[position] in top5) and (predicted_similarity[position] not in top3):
								n_top5 += 1

							# for research purposes only. Used to evaluate why mismatches happen with VG1
							if all_variants_names[pos_variant] == 'VG1':
								incorrect_matches_VG1.append([element[0], element[i], aux_list_of_questions[highest_match_index], highest_match])
						
						elif pre_selection_flag == 2:
							if len(predicted_similarity) >= 3:
								top3 = n_max_elements_indexes(predicted_similarity, 3)
							else:
								continue

							if len(predicted_similarity) >= 5:
								top5 = n_max_elements_indexes(predicted_similarity, 5)
							else:
								continue

							for index in top3:
								if aux_corpus_pairs[index] == element[0]:
									n_top3 += 1

							for index in top5:
								if aux_corpus_pairs[index] == element[0] and index not in top3:
									n_top5 += 1
						
						else:
							print("Pre-selection flag was assigned a wrong number. Please fix it and re-run the script.")


						'''
						top3 = n_max_elements_indexes(predicted_similarity, 3)
						top5 = n_max_elements_indexes(predicted_similarity, 5)

						current_value_top3 = n_top3

						for index in top3:
							# print(aux_list_of_questions[index])
							if aux_list_of_questions[index] == element[0]:
								# print("In top 3")
								n_top3 += 1

						for index in top5:
							# print(aux_list_of_questions[index])
							if aux_list_of_questions[index] == element[0] and current_value_top3 != n_top3:
								# print("In top 5")
								n_top5 += 1
						'''
						

				# uncomment if only pretends to evaluate one variant question of VUC
				if all_variants_names[pos_variant] == 'VUC' and use_only_one_vuc:
					break

			if len(all_variants[pos_variant]) == position + 1:
				print()

		variant_accuracy = (n_correct_matches/n_variant_questions)*100

		all_accuracies.append([all_variants_names[pos_variant], variant_accuracy, n_correct_matches, n_variant_questions, n_top3, n_top5, threshold_0, threshold_1, threshold_1_5, threshold_2, threshold_2_5, threshold_3, threshold_3_5, threshold_4])

	'''
	# used to plot the ROC curve
	y = np.array(y_true)
	scores = np.array(y_score)
	fpr, tpr, thresholds = metrics.roc_curve(y, scores)

	print("FPR: ")
	print(fpr)
	print("TPR: ")
	print(tpr)
	print("Threshholds: ")
	print(thresholds)

	# finding the optimal threshold
	optimal_idx = np.argmax(tpr - fpr)
	print(optimal_idx)
	optimal_threshold = thresholds[optimal_idx]
	print(optimal_threshold)

	# finding the N optimal thresholds
	array_of_differences = tpr-fpr
	list_of_differences = array_of_differences.tolist()

	optimal_indexes = n_max_elements_indexes(list_of_differences, 10)

	list_of_optimal_thresholds = []

	for index in optimal_indexes:
		list_of_optimal_thresholds.append(thresholds[index])

	print("The list of optimal thresholds is: ")
	print(list_of_optimal_thresholds)

	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.plot(fpr[optimal_idx], tpr[optimal_idx], color='red', marker='o', markersize=5, label='Optimal Threshold')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Similarity Thresholds ROC curve')
	plt.legend(loc="lower right")
	plt.show()
	plt.close()
	'''
	# writing VG1 mismatches to file
	with open('vg1_mismatches.txt', 'w') as mismatches_file:
		for item in incorrect_matches_VG1:
			mismatches_file.write("%s\n" % item)

	mismatches_file.close()

	total_correct_matches = 0
	total_variant_questions = 0
	total_top3 = 0
	total_top5 = 0

	for accuracy in all_accuracies:
		if accuracy[0] != "OG":
			if isinstance(accuracy[2], int):
				total_correct_matches += accuracy[2]
				total_variant_questions += accuracy[3]
				total_top3 += accuracy[4]
				total_top5 += accuracy[5]

	total_accuracy = (total_correct_matches/total_variant_questions)*100

	all_accuracies.extend([['Variant collection', 'Mean', 'Total Top3', 'Total Top5'], ["ALL", total_accuracy, total_top3, total_top5]])

	if generate_random_positions_file_flag == 1:
		with open('random_positions_VUC.txt', 'w') as random_positions_VUC_file:
		    for item in random_positions_VUC:
		        random_positions_VUC_file.write("%s\n" % item)

		random_positions_VUC_file.close()

	# writing the results to file
	write_output_file(model_name, all_accuracies, classification_flag, model.feature_selection, pre_selection_flag)

	print("Testing finished successfully")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')

