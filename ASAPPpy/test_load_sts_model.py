from xml.etree import cElementTree as ET
from sklearn.model_selection import train_test_split

import scripts.tools as tl
from sts_model import STSModel
from scripts.xml_reader import read_xml_no_attributes
from scripts.load_embeddings import load_embeddings_models

# test_file_path = 'datasets/assin/assin1/assin-ptbr-test.xml'
# test_file_path = 'datasets/assin/assin1/assin-ptpt-test.xml'
test_file_path = 'datasets/assin/assin2/assin2-blind-test.xml'

test_pairs = read_xml_no_attributes(test_file_path)
test_corpus = tl.read_corpus(test_pairs)

word2vec_model, fasttext_model, ptlkb64_model, glove300_model, numberbatch_model = load_embeddings_models()

model = STSModel()

model.load_model('model_1706_ablation_study_master')

test_lexical_features = 0
test_syntactic_features = 0
test_semantic_features = 0
test_distributional_features = 0
test_all_features = 0

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

similarity = model.predict_similarity(test_features)

# write output
tree = ET.parse(test_file_path)
root = tree.getroot()
for i in range(len(test_pairs)):
    pairs = root[i]
    pairs.set('entailment', "None")
    pairs.set('similarity', str(similarity[i]))

tree.write("assin2.xml", 'utf-8')