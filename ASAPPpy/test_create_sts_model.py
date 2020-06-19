import numpy as np
from xml.etree import cElementTree as ET
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

import scripts.tools as tl
from sts_model import STSModel
from scripts.xml_reader import read_xml, read_xml_no_attributes
from scripts.load_embeddings import load_embeddings_models

system_mode = 4
feature_selection_assin1 = 1
predict_similarity = 0

# extract labels
train_pairs = []
train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptpt-train.xml", need_labels=True))
train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptbr-train.xml", need_labels=True))

if system_mode == 2 or system_mode == 5:
    train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptpt-test.xml", need_labels=True))
    train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptbr-test.xml", need_labels=True))
if system_mode == 4 or system_mode == 5:
    train_pairs.extend(read_xml("datasets/assin/assin2/assin2-train-only.xml", need_labels=True))
    train_pairs.extend(read_xml('datasets/assin/assin2/assin2-dev.xml', need_labels=True))

train_similarity_target = np.array([pair.similarity for pair in train_pairs])

if feature_selection_assin1:
    train_pairs, dev_pairs, train_similarity_target, dev_target = train_test_split(train_pairs, train_similarity_target, test_size=0.1, random_state=42)

    test_pairs = read_xml_no_attributes('datasets/assin/assin1/assin-ptbr-test.xml')
    test_corpus = tl.read_corpus(test_pairs)
else:
    dev_pairs = read_xml('datasets/assin/assin2/assin2-dev.xml', need_labels=True)
    dev_target = np.array([pair.similarity for pair in dev_pairs])

    test_pairs = read_xml_no_attributes('datasets/assin/assin2/assin2-blind-test.xml')
    test_corpus = tl.read_corpus(test_pairs)

train_corpus = tl.read_corpus(train_pairs)
dev_corpus = tl.read_corpus(dev_pairs)

word2vec_model, fasttext_model, ptlkb64_model, glove300_model, numberbatch_model = load_embeddings_models()

model = STSModel(model_name='model_1706_ablation_study_master')

test_lexical_features = 0
test_syntactic_features = 0
test_semantic_features = 0
test_distributional_features = 0
test_all_features = 1

if test_lexical_features:
    model.extract_lexical_features(train_corpus, 1, 1, 1, 1, 1, 1)
    print(model.lexical_features)

if test_syntactic_features:
    # function is not finished
    model.extract_syntactic_features(train_corpus, 1, 1)
    print(model.syntactic_features)

if test_semantic_features:
    model.extract_semantic_features(train_corpus, 1, 1)
    print(model.semantic_features)

if test_distributional_features:
    # function is not correct
    model.extract_distributional_features(train_corpus, 1)
    print(model.distributional_features)

if test_all_features:
    train_features = model.extract_multiple_features(train_corpus, 0, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb_mdl=ptlkb64_model, glove_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

    dev_features = model.extract_multiple_features(dev_corpus, 0, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb_mdl=ptlkb64_model, glove_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

    test_features = model.extract_multiple_features(test_corpus, 0, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb_mdl=ptlkb64_model, glove_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

regressor = SVR(gamma='scale', C=10.0, kernel='rbf')

if predict_similarity:
    similarity = model.run_model(regressor, train_features, train_similarity_target, 1, dev_features, dev_target, test_features)

    # write output
    tree = ET.parse('datasets/assin/assin1/assin-ptbr-test.xml')
    root = tree.getroot()
    for i in range(len(test_pairs)):
        pairs = root[i]
        pairs.set('entailment', "None")
        pairs.set('similarity', str(similarity[i]))

    tree.write("test.xml", 'utf-8')
else:
    similarity = model.run_model(regressor, train_features, train_similarity_target, 1, dev_features, dev_target)

print(model.number_features)

print(model.used_features)

model.save_model()
