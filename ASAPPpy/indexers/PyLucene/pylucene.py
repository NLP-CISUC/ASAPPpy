import os
import lucene
import time

from java.io import File
from java.nio.file import Paths
from org.apache.pylucene.analysis import PythonAnalyzer
from org.apache.lucene.analysis import LowerCaseFilter, StopFilter
from org.apache.lucene.analysis.standard import StandardAnalyzer, StandardTokenizer
from org.apache.lucene.analysis.pt import PortugueseAnalyzer, PortugueseLightStemFilter
from org.apache.lucene.document import Document, Field, TextField, StringField
from org.apache.lucene.search import IndexSearcher, FuzzyQuery
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory, SimpleFSDirectory
from org.apache.lucene.util import Version

from ASAPPpy import ROOT_PATH

class customPortugueseAnalyser(PythonAnalyzer):

	def createComponents(self, fieldName):
		source = StandardTokenizer()
		stream = LowerCaseFilter(source)
		stream = StopFilter(stream, PortugueseAnalyzer.getDefaultStopSet())
		stream = PortugueseLightStemFilter(stream)

		return self.TokenStreamComponents(source, stream)

	def initReader(self, fieldName, reader):
		return reader

def l_indexer(directory, load_path):
	lucene.initVM()

	# index_dir = SimpleFSDirectory(File(directory))
	index_dir = FSDirectory.open(Paths.get(directory))
	writer_config = IndexWriterConfig(PortugueseAnalyzer())
	# writer_config = IndexWriterConfig(customPortugueseAnalyser())
	writer = IndexWriter(index_dir, writer_config)

	with open(load_path) as subtles_file:
		subtles_corpus = subtles_file.read().splitlines()

	for i in range(0, len(subtles_corpus), 2):
		doc = Document()
		doc.add(Field("question", subtles_corpus[i], StringField.TYPE_STORED))
		doc.add(Field("answer", subtles_corpus[i+1], StringField.TYPE_STORED))

		writer.addDocument(doc)

	writer.close()
	print("Index successfully created!")

def l_searcher(query_string, directory, number_documents):
	lucene.initVM()

	# analyzer = StandardAnalyzer()
	reader = DirectoryReader.open(FSDirectory.open(Paths.get(directory)))
	searcher = IndexSearcher(reader)

	# Top 'n' documents as result
	topN = number_documents

	try:
		# query = QueryParser("question", analyzer).parse(query_string)
		query = FuzzyQuery(Term("question", query_string), 2)
		print("The query was: {}".format(query))

		hits = searcher.search(query, topN)

		print("The hits were: ")

		options = []
		options_answers = []

		# print(hits.totalHits)

		for hit in hits.scoreDocs:
			print(hit.doc)
			# print(hit.score, hit.doc, hit.toString())
			doc = searcher.doc(hit.doc)
			options_answers.append(doc.get("answer"))
			options.append(doc.get("question"))
			# print(doc.get("answer"))

		return options, options_answers
	except IndexError:
		return None

if __name__ == '__main__':
	# load_path = os.path.join(ROOT_PATH, 'datasets', 'SubtleCorpusPTEN', 'por', 'corpus0sDialogues_clean_removed_entities.txt')
	# directory = os.path.join('indexes', 'Subtle_removed_entities_pylucene')

	load_path = os.path.join(ROOT_PATH, 'datasets', 'faqs_questions_pylucene_no_duplicates_16.11.txt')
	directory = os.path.join('indexes', 'FAQs_portuguese_analyser_16.11')

	start_time = time.time()
	print("Started creating the index")

	l_indexer(directory, load_path)

	print("Finished creating index successfully")
	print("--- %s seconds ---" %(time.time() - start_time))
	print('\a')

	# l = l_searcher("Olá?", directory, 10)

	