import numpy as np
from xml.etree import cElementTree as ET
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import scripts.tools as tl
from sts_model import STSModel
from scripts.xml_reader import read_xml, read_xml_no_attributes
from scripts.load_embeddings import load_embeddings_models

system_mode = 4
feature_selection_flag = 0
manual_feature_selection_flag = 1
predict_similarity = 1

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

if feature_selection_flag:

    feature_selection_assin1 = 1

    if feature_selection_assin1:
        train_pairs, dev_pairs, train_similarity_target, dev_target = train_test_split(train_pairs, train_similarity_target, test_size=0.1, random_state=42)
    else:
        dev_pairs = read_xml('datasets/assin/assin2/assin2-dev.xml', need_labels=True)
        dev_target = np.array([pair.similarity for pair in dev_pairs])

    dev_corpus = tl.read_corpus(dev_pairs)

if predict_similarity:
    test_dataset_flag = 0

    if test_dataset_flag == 0:
        test_path = 'datasets/assin/assin1/assin-ptbr-test.xml'

    if test_dataset_flag == 1:
        test_path = 'datasets/assin/assin1/assin-ptpt-test.xml'

    if test_dataset_flag == 2:
        test_path = 'datasets/assin/assin2/assin2-blind-test.xml'

    test_pairs = read_xml_no_attributes(test_path)
    test_corpus = tl.read_corpus(test_pairs)

train_corpus = tl.read_corpus(train_pairs)

word2vec_model, fasttext_model, ptlkb64_model, glove300_model, numberbatch_model = load_embeddings_models()

model = STSModel(model_name='model_2206_RFR_R_pos_adv-depedency_parsing-word2vec-ptlkb-numberbatch')

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

if manual_feature_selection_flag:
    selected_features = {'wn_jaccard_1': 1, 'wn_jaccard_2': 0, 'wn_jaccard_3': 0, 'wn_dice_1': 1, 'wn_dice_2': 1, 'wn_dice_3': 0, 'wn_overlap_1': 1, 'wn_overlap_2': 1, 'wn_overlap_3': 0, 'cn_jaccard_2': 1, 'cn_jaccard_3': 1, 'cn_jaccard_4': 1, 'cn_dice_2': 1, 'cn_dice_3': 1, 'cn_dice_4': 1, 'cn_overlap_2': 1, 'cn_overlap_3': 1, 'cn_overlap_4': 1, 'pos_adj': 0, 'pos_adv': 0, 'pos_art': 0, 'pos_conj-c': 0, 'pos_conj-s': 0, 'pos_intj': 0, 'pos_n': 0, 'pos_n-adj': 0, 'pos_num': 0, 'pos_pron-det': 0, 'pos_pron-indp': 0, 'pos_pron-pers': 0, 'pos_prop': 0, 'pos_prp': 0, 'pos_punc': 0, 'pos_v-fin': 0, 'pos_v-ger': 0, 'pos_v-inf': 0, 'pos_v-pcp': 0, 'dependency_parsing': 0, 'sr_antonyms': 0, 'sr_synonyms': 0, 'sr_hyperonyms': 0, 'sr_other': 0, 'all_ne': 0, 'ne_B-ABSTRACCAO': 0, 'ne_B-ACONTECIMENTO': 0, 'ne_B-COISA': 0, 'ne_B-LOCAL': 0, 'ne_B-OBRA': 0, 'ne_B-ORGANIZACAO': 0, 'ne_B-OUTRO': 0, 'ne_B-PESSOA': 0, 'ne_B-TEMPO': 0, 'ne_B-VALOR': 0, 'word2vec': 1, 'word2vec_tfidf': 1, 'fasttext': 1, 'fasttext_tfidf': 1, 'ptlkb': 0, 'ptlkb_tfidf': 0, 'glove': 1, 'glove_tfidf': 1, 'numberbatch': 0, 'numberbatch_tfidf': 0, 'tfidf': 1}

if test_all_features:
    #if feature_selection_flag:
    if manual_feature_selection_flag:
        train_features = model.extract_multiple_features(train_corpus, 0, manual_feature_selection=selected_features, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb_mdl=ptlkb64_model, glove_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

        dev_features = None
        #dev_features = model.extract_multiple_features(dev_corpus, 0, manual_feature_selection=selected_features, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb_mdl=ptlkb64_model, glove_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

        test_features = model.extract_multiple_features(test_corpus, 0, manual_feature_selection=selected_features, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb_mdl=ptlkb64_model, glove_mdl=glove300_model, numberbatch_mdl=numberbatch_model)
    else:
        train_features = model.extract_multiple_features(train_corpus, 0, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb_mdl=ptlkb64_model, glove_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

        test_features = model.extract_multiple_features(test_corpus, 0, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb_mdl=ptlkb64_model, glove_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

#regressor = SVR(gamma='scale', C=10.0, kernel='rbf')
#regressor = GradientBoostingRegressor()
regressor = RandomForestRegressor(n_estimators=100)

if predict_similarity:
    if feature_selection_flag:
        similarity = model.run_model(regressor, train_features, train_similarity_target, test_features=test_features)
    else:
        similarity = model.run_model(regressor, train_features, train_similarity_target, test_features=test_features)

    # write output
    tree = ET.parse(test_path)
    root = tree.getroot()
    for i in range(len(test_pairs)):
        pairs = root[i]
        pairs.set('entailment', "None")
        pairs.set('similarity', str(similarity[i]))

    # tree.write("test.xml", 'utf-8')
    tree.write("assin1-ptbr.xml", 'utf-8')
else:
    similarity = model.run_model(regressor, train_features, train_similarity_target, 1, dev_features, dev_target)

print(model.number_features)

print(model.used_features)

model.save_model()
