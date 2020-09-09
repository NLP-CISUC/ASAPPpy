import sys
import tmp_feature_extraction as fe

from importlib import reload

word2vec_model = None
fasttext_model = None
ptlkb64_model = None
glove300_model = None
numberbatch_model = None

if __name__ == "__main__":
    models_loaded = 0
    while True:
        if models_loaded == 0:
            word2vec_model, fasttext_model, ptlkb64_model, glove300_model, numberbatch_model = fe.load_embeddings_models()
            models_loaded = 1
        fe.run_feature_extraction(word2vec_model=word2vec_model, fasttext_model=fasttext_model, ptlkb64_model=ptlkb64_model, glove300_model=glove300_model, numberbatch_model=numberbatch_model)
        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()
        reload(fe)
        