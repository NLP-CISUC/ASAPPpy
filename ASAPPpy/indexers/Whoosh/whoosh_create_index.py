import os
import sys
import time
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer, CharsetFilter, NgramFilter
from whoosh.support.charset import accent_map
from ASAPPpy import ROOT_PATH

def createSearchableData(directory, load_path):
    '''
    Schema definition: title(name of file), path(as ID), content(indexed
    but not stored),textdata (stored text content)
    '''
    # the call of the StemmingAnalyzer had to be changed in the whoosh directory to support the portuguese language
    my_analyzer = StemmingAnalyzer() | CharsetFilter(accent_map) | NgramFilter(minsize=2, maxsize=4)
    schema = Schema(question=TEXT(analyzer=my_analyzer, stored=True), response=TEXT(analyzer=my_analyzer, stored=True))
    # schema = Schema(question=TEXT(stored=True), response=TEXT(stored=True))
    schema.cachesize = -1

    if not os.path.exists(directory):

        # makedirs is used to create directories with subdirectories in it
        os.makedirs(directory)

    # Creating a index writer to add document as per schema
    ix = create_in(directory, schema)
    writer = ix.writer(limitmb=1024)

    with open(load_path) as subtles_file:
        subtles_corpus = subtles_file.read().splitlines()

    for i in range(0, len(subtles_corpus), 2):
        writer.add_document(question=subtles_corpus[i], response=subtles_corpus[i+1])
    writer.commit()

if __name__ == '__main__':

    directory = os.path.join('indexes', 'Subtle_removed_entities')
    load_path = os.path.join(ROOT_PATH, 'datasets', 'SubtleCorpusPTEN', 'por', 'corpus0sDialogues_clean_removed_entities.txt')

    start_time = time.time()
    print("Started creating the index")

    createSearchableData(directory, load_path)

    print("Finished creating index successfully")
    print("--- %s seconds ---" %(time.time() - start_time))
    print('\a')
