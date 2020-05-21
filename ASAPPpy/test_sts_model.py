import numpy as np
import tools as tl
from sts_model import STSModel
from assin.assineval.commons import read_xml
from sklearn.svm import SVR

system_mode = 4

# Flag to indicate if the extracted features should be written to a file (1) or not (0)
features_to_file_flag = 0

# extract labels
train_pairs = []
# train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptpt-train.xml", need_labels=True))
# train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptbr-train.xml", need_labels=True))

if system_mode == 2 or system_mode == 5:
    train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptpt-test.xml", need_labels=True))
    train_pairs.extend(read_xml("datasets/assin/assin1/assin-ptbr-test.xml", need_labels=True))
if system_mode == 4 or system_mode == 5:
    train_pairs.extend(read_xml("datasets/assin/assin2/assin2-train-only.xml", need_labels=True))

train_similarity_target = np.array([pair.similarity for pair in train_pairs])

# extract training features
train_corpus = tl.read_corpus(train_pairs)

# tl.write_data_to_file(train_corpus, "finetune.train.raw")
# print("Wrote training corpus")

# preprocessing(text, tokenization=0, rm_stopwords=0, numbers_to_text=0, to_tfidf=0)
# preprocessed_train_corpus = tl.preprocessing(train_corpus, 0, 0, 0, 0)

new_model = STSModel()

test_lexical_features = 0
test_syntactic_features = 0
test_semantic_features = 0
test_distributional_features = 0
test_all_features = 1

if test_lexical_features:
    new_model.extract_lexical_features(train_corpus, 1, 1, 1, 1, 1, 1)
    print(new_model.lexical_features)

if test_syntactic_features:
    # function is not finished
    new_model.extract_syntactic_features(train_corpus, 1, 1)
    print(new_model.syntactic_features)

if test_semantic_features:
    new_model.extract_semantic_features(train_corpus, 1, 1)
    print(new_model.semantic_features)

if test_distributional_features:
    # function is not correct
    new_model.extract_distributional_features(train_corpus, 1)
    print(new_model.distributional_features)

if test_all_features:
    new_model.extract_all_features(train_corpus)
    print(new_model.all_features)

regressor = SVR(gamma='scale', C=10.0, kernel='rbf')

similarity = new_model.run_model(0, regressor, train_similarity_target)

print(similarity)

new_model.save_model()

