import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
# from assin.assineval.commons import read_xml

# import tools as tl

from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder()
tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
vectors = elmo.embed_sentence(tokens)

print(vectors)

'''
def convert_text_to_embedding(tokenizer, model, sentence):
	marked_sentence = "[CLS] " + sentence + " [SEP] "

	tokenized_sentence = tokenizer.tokenize(marked_sentence)

	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)

	segments_ids = [1] * len(tokenized_sentence)

	# Convert inputs to PyTorch tensors
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])

	# Predict hidden states features for each layer
	with torch.no_grad():
		encoded_layers, _ = model(tokens_tensor, segments_tensors)

	sentence_embedding = torch.mean(encoded_layers[11], 1)

	return sentence_embedding

def elmo_model(corpus):
	# Load pre-trained model tokenizer (vocabulary)
	# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

	# Load pre-trained model (weights)
	# model = BertModel.from_pretrained('bert-base-uncased')
	model = BertModel.from_pretrained('bert-base-multilingual-cased')

	# Put the model in "evaluation" mode, meaning feed-forward operation.
	model.eval()

	bert_embeddings = []

	for i in range(0, len(corpus), 2):
		print("%d/%d" %(i, len(corpus)))
		sentence_embedding_1 = convert_text_to_embedding(tokenizer, model, corpus[i])
		sentence_embedding_2 = convert_text_to_embedding(tokenizer, model, corpus[i+1])

		similarity = cosine_similarity(sentence_embedding_1, sentence_embedding_2)
		print(similarity[0][0])
		bert_embeddings.append(similarity[0][0])

	return bert_embeddings

if __name__ == '__main__':
	system_mode = 5
	run_pipeline = 1

	# extract labels
	train_pairs = []
	train_pairs.extend(read_xml("assin-ptpt-train.xml", need_labels=True))
	train_pairs.extend(read_xml("assin-ptbr-train.xml", need_labels=True))

	if system_mode == 2 or system_mode == 5:
		train_pairs.extend(read_xml("assin-ptpt-test.xml", need_labels=True))
		train_pairs.extend(read_xml("assin-ptbr-test.xml", need_labels=True))
	if system_mode == 4 or system_mode == 5:
		train_pairs.extend(read_xml("assin2-train-only.xml", need_labels=True))

	# extract training features
	train_corpus = tl.read_corpus(train_pairs)

	embeddings = elmo_model(train_corpus)
	print(embeddings)

'''