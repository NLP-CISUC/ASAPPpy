import csv
from xml.etree import cElementTree as ET

def combine_similarity_entailment(filename_in, filename_out):
	with open(filename_in) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		list_csv_reader = list(csv_reader)

	tree = ET.parse(filename_out)
	root = tree.getroot()
	for i in range(len(list_csv_reader)):
		pairs = root[i]
		pairs.set('entailment', str(list_csv_reader[i][1]))

	tree.write(filename_out, 'utf-8')

if __name__ == '__main__':
	file_in = 'run3_ASSIN2_test_ASAPPpy_entailment.csv'
	file_out = 'assin1_all_assin2_competition.xml'

	combine_similarity_entailment(file_in, file_out)
