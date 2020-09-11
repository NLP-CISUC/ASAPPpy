import os

from xml.etree import cElementTree as ET

from ASAPPpy import ROOT_PATH
from ASAPPpy.scripts.tools import read_corpus
from ASAPPpy.sts_model import STSModel
from ASAPPpy.scripts.xml_reader import read_xml_no_attributes
from ASAPPpy.scripts.load_embeddings import load_embeddings_models

#test_file_path = os.path.join(ROOT_PATH, 'datasets', 'assin', 'assin1', 'assin-ptbr-test.xml')
#test_file_path = os.path.join(ROOT_PATH, 'datasets', 'assin', 'assin1', 'assin-ptpt-test.xml')
test_file_path = os.path.join(ROOT_PATH, 'datasets', 'assin', 'assin2', 'assin2-blind-test.xml')

test_pairs = read_xml_no_attributes(test_file_path)
test_corpus = read_corpus(test_pairs)

word2vec_model, fasttext_model, ptlkb64_model, glove300_model, numberbatch_model = load_embeddings_models()

model = STSModel()

model.load_model('model_2206_RFR_R_pos_adv-depedency_parsing-word2vec-ptlkb-numberbatch')

test_features = model.extract_multiple_features(corpus=test_corpus, to_store=0, word2vec_mdl=word2vec_model, fasttext_mdl=fasttext_model, ptlkb_mdl=ptlkb64_model, glove_mdl=glove300_model, numberbatch_mdl=numberbatch_model)

similarity = model.predict_similarity(test_features)

# write output
tree = ET.parse(test_file_path)
root = tree.getroot()
for i in range(len(test_pairs)):
	pairs = root[i]
	pairs.set('entailment', "None")
	pairs.set('similarity', str(similarity[i]))

tree.write("assin2.xml", 'utf-8')