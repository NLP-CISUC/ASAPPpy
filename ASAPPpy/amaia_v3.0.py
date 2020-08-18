import os
import copy
import re
import asyncio
import slack
import pandas as pd

from gensim.parsing.preprocessing import strip_non_alphanum
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from string import punctuation

from ASAPPpy.resources.resources import read_class_set
from ASAPPpy.feature_extraction import load_embeddings_models
import ASAPPpy.indexers.Whoosh.whoosh_make_query as qwi
from ASAPPpy.classifiers.svm_restantes_classes import corre_para_testes_restantes
from ASAPPpy.classifiers.svm_binaria import corre_para_frase
from ASAPPpy.sts_model import STSModel

# constants
SLACK_SIGNING_SECRET = os.environ['SLACK_SIGNING_SECRET']
SLACK_BOT_TOKEN = os.environ['COBAIA_BOT_TOKEN']

# cobaiabot's user ID in Slack: value is assigned after the bot starts up
bot_id = None

# constants
MENTION_REGEX = "^<@(|[WU].+?)>(.*)"

word2vec_model, fasttext_model, ptlkb64_model, glove300_model, numberbatch_model = load_embeddings_models()

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

def chatbot_interface(interaction, word2vec_model, fasttext_model, ptlkb64_model, glove300_model, numberbatch_model):
	""" Function used to run the chatbot interface """
	# Flag to indicate if classification should be used (1) or not (0)
	classification_flag = 1

	# Flag to indicate if the binary classifier should be used (1) or not (0)
	binary_classifier_flag = 1

	# choose if stopwords should be removed from the user interaction
	process_interaction_toggle = 0

	# choose if pre-selection should not be used (0), used with word embeddings (1) or used with Whoosh (2)
	pre_selection_toggle = 2

	# parameters used to tune the selection of more than one response
	sr_alpha = 0.1
	sr_beta = 3

	# TODO: The STS model class can't perform feature selection but can use a model that uses it already.

	# location of the STS model
	model = STSModel()
	model.load_model('model_R_pos_adv-dependency_parsing-word2vec-ptlkb-numberbatch')

	if classification_flag:
		print("The classifier is being used.")

		# read the different class sets
		class_1, class_2, class_3 = read_class_set()

		# transform the class sets from lists to dataframes
		class_1_df = pd.DataFrame(class_1, columns=['text'])
		class_2_df = pd.DataFrame(class_2, columns=['text'])
		class_3_df = pd.DataFrame(class_3, columns=['text'])

	faqs_variants_load_path = os.path.join('datasets', 'AIA-BDE_v2.0.txt')

	with open(faqs_variants_load_path) as faqs_file:
		faqs_variants_corpus = faqs_file.read().splitlines()

	faqs_file.close()

	faqs_variants_corpus = [line.replace('\t', '') for line in faqs_variants_corpus]
	faqs_variants_corpus = [line.split(':', 1) for line in faqs_variants_corpus]

	# add the original question to a different list to improve the conversational presentation of a response
	position = 0
	faqs_variants_questions = []

	for element in faqs_variants_corpus:
		if element[0] == 'P' and element[1] not in faqs_variants_questions:
			faqs_variants_questions.append(element[1])
			position += 1

	# add the original question to a different list to improve the conversational presentation of a response
	position = 0
	faqs_variants_answers = []

	for element in faqs_variants_corpus:
		if element[0] == 'R' and element[1] not in faqs_variants_answers:
			faqs_variants_answers.append(element[1])
			position += 1

	faqs_variants_corpus = [line for line in faqs_variants_corpus if len(line) == 2 and line[1]]
	faqs_variants_corpus = [[line[0], strip_non_alphanum(line[1])] if line[0] != 'R' else [line[0], line[1]] for line in faqs_variants_corpus]
	faqs_variants_corpus = [[line[0].rstrip(), line[1].rstrip()] if line[0] != 'R' else [line[0], line[1]] for line in faqs_variants_corpus]
	faqs_variants_corpus = [[line[0], strip_multiple_whitespaces(line[1])] if line[0] != 'R' else [line[0], line[1]] for line in faqs_variants_corpus]
	faqs_variants_corpus = [[line[0], line[1].lower()] if line[0] != 'R' else [line[0], line[1]] for line in faqs_variants_corpus]

	position = 0
	corpus = []

	for element in faqs_variants_corpus:
		if element[0] == 'P':
			corpus.append([element[1]])

		if element[0] == 'R':
			corpus[position].extend([element[1]])

			position += 1

	aux_list_of_questions = [phrases[0] for phrases in corpus]
	aux_df = pd.DataFrame(faqs_variants_questions, columns=['text'])

	# remove duplicate sentences from the aux_list_of_questions in order for Whoosh to work.
	clean_aux_list_of_questions = []

	for pair in corpus:
		if pair not in clean_aux_list_of_questions:
			clean_aux_list_of_questions.append(pair)

	if process_interaction_toggle:
		print("The original sentence was: {}".format(interaction))
		stp = set(stopwords.words('portuguese') + list(punctuation))
		interaction = ' '.join([word for word in interaction.split(' ') if word not in stp])
		print("The sentenced after removing stopwords and punctuation: {}".format(interaction))

	unprocessed_corpus = []

	if classification_flag:
		# apply the classifier before using the STS model
		if binary_classifier_flag:
			predicted_class = corre_para_frase(interaction)
			print("Saí daqui")
			if predicted_class == 0:
				print("The provided interaction is out of domain!\n")
		else:
			predicted_class = corre_para_testes_restantes([interaction])

			if predicted_class == 1:
				print("The provided interaction belongs to class 1!\n")
				aux_df = class_1_df
				aux_list_of_questions = class_1
			elif predicted_class == 2:
				print("The provided interaction belongs to class 2!\n")
				aux_df = class_2_df
				aux_list_of_questions = class_2
			elif predicted_class == 3:
				print("The provided interaction belongs to class 3!\n")
				aux_df = class_3_df
				aux_list_of_questions = class_3
			else:
				print("The provided interaction is out of domain!\n")

	if predicted_class == 1:
		if 'response' not in aux_df:
			aux_df.insert(1, 'response', interaction)
		else:
			aux_df['response'] = interaction

		if pre_selection_toggle != 2:
			for j in range(len(faqs_variants_questions)):
				if pre_selection_toggle == 1:
					unprocessed_corpus.append([faqs_variants_questions[j], interaction])
				else:
					unprocessed_corpus.extend([faqs_variants_questions[j], interaction])

		if pre_selection_toggle == 1:
			corpus_pairs, indexes = pre_selection(unprocessed_corpus, fasttext_model, position)

			if corpus_pairs is None:
				index_path = os.path.join('indexers', 'Whoosh', 'indexes', 'cobaia_chitchat_v1.5')

				query_response = qwi.query_indexer(interaction, index_path)

				if (query_response[0] is None) or (not query_response[0]):
					response = "Desculpe, não percebi, pode colocar a sua questão de outra forma?"
					return response
				else:
					return response

			selected_aux_df = aux_df.iloc[indexes]
			selected_aux_df = selected_aux_df.reset_index(drop=True)
		else:
			if pre_selection_toggle == 2:
				pre_selection_index_path = os.path.join('indexers', 'Whoosh', 'indexes', 'FAQs_no_analyser_AIA-BDE_v2.0')

				query_response = qwi.query_indexer(interaction, pre_selection_index_path)
				options_docnumbers = query_response[2]

				if len(options_docnumbers) == 0:
					response = "Desculpe, não percebi, pode colocar a sua questão de outra forma?"
					return response
				else:
					possible_variants_questions = []
					possible_variants_answers = []

					for pos, elem in enumerate(options_docnumbers):
						unprocessed_corpus.extend([faqs_variants_questions[elem], interaction])
						possible_variants_questions.append(faqs_variants_questions[elem])
						possible_variants_answers.append(faqs_variants_answers[elem])

			corpus_pairs = unprocessed_corpus
			selected_aux_df = aux_df

		element_features = model.extract_multiple_features(corpus_pairs, 0, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb_mdl=ptlkb64_model, glove_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

		predicted_similarity = model.predict_similarity(element_features)
		predicted_similarity = predicted_similarity.tolist()

		highest_match = max(predicted_similarity)

		selectable_range = (max(predicted_similarity)-min(predicted_similarity)) * sr_alpha

		if sr_beta > len(predicted_similarity):
			tmp_sr_beta = len(predicted_similarity)
			sr_beta_range = tmp_sr_beta
			possible_matches = n_max_elements(predicted_similarity, tmp_sr_beta)
		else:
			sr_beta_range = sr_beta
			possible_matches = n_max_elements(predicted_similarity, sr_beta)

		highest_match_index = predicted_similarity.index(max(predicted_similarity))

		if pre_selection_toggle == 2:
			response = ("Se a sua pergunta foi: %s \nR: %s\n" % (possible_variants_questions[highest_match_index], possible_variants_answers[highest_match_index]))

			for i in range(1, sr_beta_range):
				if abs(highest_match-possible_matches[i]) <= selectable_range:
					response += ("Também poderá estar interessado em: %s\nR: %s\n" % (possible_variants_questions[predicted_similarity.index(possible_matches[i])], possible_variants_answers[predicted_similarity.index(possible_matches[i])]))

			return response
		else:
			#should be index 1, for testing purposes it is 0
			response = ("Se a sua pergunta foi: %s \nR: %s\n" % (faqs_variants_questions[highest_match_index], faqs_variants_answers[highest_match_index]))

			for i in range(1, sr_beta):
				if abs(highest_match-possible_matches[i]) <= selectable_range:
					response += ("Também poderá estar interessado em: %s\nR: %s\n" % (faqs_variants_questions[predicted_similarity.index(possible_matches[i])], faqs_variants_answers[predicted_similarity.index(possible_matches[i])]))

			return response
	else:
		# the query search will return a list of phrases with the highest matches, which will be used with the similarity model in order to evaluate which answer should be returned to the user
		index_path = os.path.join('indexers', 'Whoosh', 'indexes', 'cobaia_chitchat_v1.5')

		query_response = qwi.query_indexer(interaction, index_path, 1)
		print(query_response[0])
		print(query_response[1])
		if (query_response[0] is None) or (not query_response[0]):
			response = "Desculpe, não percebi, pode colocar a sua questão de outra forma?"
			return response
		else:
			'''
			unprocessed_answers = []
			aux_qwi = pd.DataFrame(query_response[0], columns=['text'])

			if 'response' not in aux_qwi:
				aux_qwi.insert(1, 'response', interaction)
			else:
				aux_qwi['response'] = interaction

			for k in range(len(query_response[0])):
				unprocessed_answers.extend([faqs_variants_questions[k], interaction])

			# element_features_qwi = extract_features(0, unprocessed_answers, aux_qwi, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb64_mdl=ptlkb64_model, glove300_mdl=glove300_model, numberbatch_mdl=numberbatch_model, f_selection=converted_mask)

			element_features_qwi = model.extract_multiple_features(unprocessed_answers, 0, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb_mdl=ptlkb64_model, glove_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

			predicted_similarity_qwi = model.predict_similarity(element_features_qwi)
			predicted_similarity_qwi = predicted_similarity_qwi.tolist()
			print(predicted_similarity_qwi)

			highest_match_index_qwi = predicted_similarity_qwi.index(max(predicted_similarity_qwi))

			return query_response[1][highest_match_index_qwi]
			'''
			return query_response[1][0]

@slack.RTMClient.run_on(event='message')
async def handle_message(**payload):

	data = payload['data']

	# get the text of the message in order to search for a response
	text = data.get('text')
	# get the channel in which the message was sent
	channel = data.get('channel')

	# if the first letter in the channel name is a D, it means that is was a direct message to the chatbot
	if channel[0] == "D":
		if data.get('subtype') is None and text:
			message = chatbot_interface(text, word2vec_model, fasttext_model, ptlkb64_model, glove300_model, numberbatch_model)

			webclient = payload['web_client']
			await webclient.chat_postMessage(
				channel=channel,
				text=message
			)
	else:
		# if not a direct message, check whether or not the chatbot was mentioned in the conversation
		user_id, text = parse_direct_mention(text)

		if user_id == bot_id:
			if data.get('subtype') is None and text:
				message = chatbot_interface(text, word2vec_model, fasttext_model, ptlkb64_model, glove300_model, numberbatch_model)

				webclient = payload['web_client']
				await webclient.chat_postMessage(
					channel=channel,
					text=message
				)

def parse_direct_mention(message_text):
	"""
		Finds a direct mention (a mention that is at the beginning) in message text
		and returns the user ID which was mentioned. If there is no direct mention, returns None
	"""
	matches = re.search(MENTION_REGEX, message_text)
	# the first group contains the username, the second group contains the remaining message
	return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

if __name__ == "__main__":
	# TODO (Future Work): Verify if it is possible to use @mentions in a more elegant way
	slack_web_client = slack.WebClient(SLACK_BOT_TOKEN)

	# extract the bot_id in order to check if it is mentioned in a channel conversation
	if bot_id is None:
		bot_id = slack_web_client.api_call("auth.test")["user_id"]

	loop = asyncio.new_event_loop()
	asyncio.set_event_loop(loop)
	slack_client = slack.RTMClient(
		token=SLACK_BOT_TOKEN, run_async=True, loop=loop
	)
	loop.run_until_complete(slack_client.start())
