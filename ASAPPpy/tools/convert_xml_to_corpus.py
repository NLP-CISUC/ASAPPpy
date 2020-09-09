import os

from assin.assineval.commons import read_xml
from gensim.parsing.preprocessing import strip_multiple_whitespaces

def build_training_corpus(filenames):
    corpus = []

    for filename in filenames:
        train_list = read_xml(filename, 1)

        for pair in train_list:
            corpus += [strip_multiple_whitespaces(pair.t)]
            corpus += [strip_multiple_whitespaces(pair.h)]

    return corpus

def write_training_corpus(data, filename):
	""" Function used to debug the corpus state during preprocessing """
	with open(filename, 'w') as f:
		for item in data:
			f.write("%s\n" % item)

if __name__ == "__main__":
	filenames_list = []

	corpus_path_1 = os.path.join('assin-ptpt-train.xml')
	corpus_path_2 = os.path.join('assin-ptbr-train.xml')
	corpus_path_3 = os.path.join('assin-ptpt-test.xml')
	corpus_path_4 = os.path.join('assin-ptbr-test.xml')
	corpus_path_5 = os.path.join('assin2-train.xml')

	filenames_list.extend([corpus_path_1, corpus_path_2, corpus_path_3, corpus_path_4, corpus_path_5])

	train_corpus = build_training_corpus(filenames_list)

	write_training_corpus(train_corpus, "complete_assin_training_assin1_testing_corpus.txt")
	
