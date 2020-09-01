import os
import numpy as np
from joblib import dump, load

from ASAPPpy import ROOT_PATH
from .feature_engineering.lexical_features import create_word_ngrams, create_multiple_word_ngrams, create_character_ngrams, create_multiple_character_ngrams, compute_jaccard, compute_dice, compute_overlap, NG
from .feature_engineering.syntactic_features import compute_pos, dependency_parsing
from .feature_engineering.semantic_features import compute_ner, compute_semantic_relations
from .feature_selection.feature_selection import feature_selection
from .models.word2vec.word2vec import word2vec_model
from .models.fastText.fasttext import fasttext_model
from .models.ontoPT.ptlkb import ptlkb_model
# TODO: rename and relocate this file:
from .load_embeddings import word_embeddings_model
from .scripts.tools import build_sentences_from_tokens, compute_tfidf_matrix, preprocessing

from NLPyPort.FullPipeline import new_full_pipe

class STSModel():
    '''Semantic Textual Similarity Model.

    Add longer class information here....

    Attributes:
        model_name: .
        model: .
        number_features: .
        used_features: .
        lexical_features: .
        syntactic_features: .
        semantic_features: .
        distributional_features: .
        all_features: .
        feature_selection: .
    '''

    def __init__(self, model_name='default_model_name', model=None, number_features=0):
        self.model_name = model_name
        self.model = model
        self.number_features = number_features
        self.used_features = {}
        self.lexical_features = None
        self.syntactic_features = None
        self.semantic_features = None
        self.distributional_features = None
        self.all_features = None
        self.feature_selection = 0

    # TODO (TESTING): there is no option to load a model that uses feature selection.
    def extract_lexical_features(self, corpus, wn_jaccard, wn_dice, wn_overlap, cn_jaccard, cn_dice, cn_overlap):
        '''
        Parameters
        ----------


        '''

        if corpus is None:
            print("Argument corpus is empty, returning None")
            return
        else:
            preprocessed_corpus = preprocessing(corpus, 0, 0, 0, 0)

        # create word ngrams of different sizes
        if wn_jaccard or wn_dice or wn_overlap:
            word_ngrams_1 = create_word_ngrams(preprocessed_corpus, 1)
            word_ngrams_2 = create_word_ngrams(preprocessed_corpus, 2)
            word_ngrams_3 = create_word_ngrams(preprocessed_corpus, 3)

        # compute distance coefficients for these ngrams
        if wn_jaccard:
            wn_jaccard_1 = compute_jaccard(word_ngrams_1)
            wn_jaccard_2 = compute_jaccard(word_ngrams_2)
            wn_jaccard_3 = compute_jaccard(word_ngrams_3)
            self.used_features.update({'wn_jaccard_1': 1, 'wn_jaccard_2': 1, 'wn_jaccard_3': 1})
        else:
            self.used_features.update({'wn_jaccard_1': 0, 'wn_jaccard_2': 0, 'wn_jaccard_3': 0})

        if wn_dice:
            wn_dice_1 = compute_dice(word_ngrams_1)
            wn_dice_2 = compute_dice(word_ngrams_2)
            wn_dice_3 = compute_dice(word_ngrams_3)
            self.used_features.update({'wn_dice_1': 1, 'wn_dice_2': 1, 'wn_dice_3': 1})
        else:
            self.used_features.update({'wn_dice_1': 0, 'wn_dice_2': 0, 'wn_dice_3': 0})

        if wn_overlap:
            wn_overlap_1 = compute_overlap(word_ngrams_1)
            wn_overlap_2 = compute_overlap(word_ngrams_2)
            wn_overlap_3 = compute_overlap(word_ngrams_3)
            self.used_features.update({'wn_overlap_1': 1, 'wn_overlap_2': 1, 'wn_overlap_3': 1})
        else:
            self.used_features.update({'wn_overlap_1': 0, 'wn_overlap_2': 0, 'wn_overlap_3': 0})

        # create character ngrams of different sizes
        if cn_jaccard or cn_dice or cn_overlap:
            character_ngrams_2 = create_character_ngrams(preprocessed_corpus, 2)
            character_ngrams_3 = create_character_ngrams(preprocessed_corpus, 3)
            character_ngrams_4 = create_character_ngrams(preprocessed_corpus, 4)

        # compute distance coefficients for these ngrams
        if cn_jaccard:
            cn_jaccard_2 = compute_jaccard(character_ngrams_2)
            cn_jaccard_3 = compute_jaccard(character_ngrams_3)
            cn_jaccard_4 = compute_jaccard(character_ngrams_4)
            self.used_features.update({'cn_jaccard_2': 1, 'cn_jaccard_3': 1, 'cn_jaccard_4': 1})
        else:
            self.used_features.update({'cn_jaccard_2': 0, 'cn_jaccard_3': 0, 'cn_jaccard_4': 0})

        if cn_dice:
            cn_dice_2 = compute_dice(character_ngrams_2)
            cn_dice_3 = compute_dice(character_ngrams_3)
            cn_dice_4 = compute_dice(character_ngrams_4)
            self.used_features.update({'cn_dice_2': 1, 'cn_dice_3': 1, 'cn_dice_4': 1})
        else:
            self.used_features.update({'cn_dice_2': 0, 'cn_dice_3': 0, 'cn_dice_4': 0})

        if cn_overlap:
            cn_overlap_2 = compute_overlap(character_ngrams_2)
            cn_overlap_3 = compute_overlap(character_ngrams_3)
            cn_overlap_4 = compute_overlap(character_ngrams_4)
            self.used_features.update({'cn_overlap_2': 1, 'cn_overlap_3': 1, 'cn_overlap_4': 1})
        else:
            self.used_features.update({'cn_overlap_2': 0, 'cn_overlap_3': 0, 'cn_overlap_4': 0})

        tmp_lexical_features = []

        for pair in range(len(corpus)//2):
            pair_features = []

            if wn_jaccard:
                pair_features.append(wn_jaccard_1['jaccard'][pair])
                pair_features.append(wn_jaccard_2['jaccard'][pair])
                pair_features.append(wn_jaccard_3['jaccard'][pair])
            else:
                pair_features.extend([0]*3)

            if wn_dice:
                pair_features.append(wn_dice_1['dice'][pair])
                pair_features.append(wn_dice_2['dice'][pair])
                pair_features.append(wn_dice_3['dice'][pair])
            else:
                pair_features.extend([0]*3)

            if wn_overlap:
                pair_features.append(wn_overlap_1['overlap'][pair])
                pair_features.append(wn_overlap_2['overlap'][pair])
                pair_features.append(wn_overlap_3['overlap'][pair])
            else:
                pair_features.extend([0]*3)

            if cn_jaccard:
                pair_features.append(cn_jaccard_2['jaccard'][pair])
                pair_features.append(cn_jaccard_3['jaccard'][pair])
                pair_features.append(cn_jaccard_4['jaccard'][pair])
            else:
                pair_features.extend([0]*3)

            if cn_dice:
                pair_features.append(cn_dice_2['dice'][pair])
                pair_features.append(cn_dice_3['dice'][pair])
                pair_features.append(cn_dice_4['dice'][pair])
            else:
                pair_features.extend([0]*3)

            if cn_overlap:
                pair_features.append(cn_overlap_2['overlap'][pair])
                pair_features.append(cn_overlap_3['overlap'][pair])
                pair_features.append(cn_overlap_4['overlap'][pair])
            else:
                pair_features.extend([0]*3)

            pair_features_tuple = tuple(pair_features)
            tmp_lexical_features.append(pair_features_tuple)

        self.lexical_features = np.array(tmp_lexical_features)

        return self.lexical_features

    def _extract_lexical_features(self, corpus):
        '''
        Parameters
        ----------
        When loading a previously trained model this function extracts only the features that were used by it instead of extracting them all.

        '''

        if corpus is None:
            print("Argument corpus is empty, returning None")
            return
        else:
            preprocessed_corpus = preprocessing(corpus, 0, 0, 0, 0)

        # check if any wn feature was used by the model before computing the word n_grams
        if any(key[:2] == 'wn' and value == 1 for key, value in self.used_features.items()):
            word_ngrams_1 = create_word_ngrams(preprocessed_corpus, 1)
            word_ngrams_2 = create_word_ngrams(preprocessed_corpus, 2)
            word_ngrams_3 = create_word_ngrams(preprocessed_corpus, 3)

        # compute distance coefficients for these ngrams
        if self.used_features['wn_jaccard_1'] == 1:
            wn_jaccard_1 = compute_jaccard(word_ngrams_1)
        if self.used_features['wn_jaccard_2'] == 1:
            wn_jaccard_2 = compute_jaccard(word_ngrams_2)
        if self.used_features['wn_jaccard_3'] == 1:
            wn_jaccard_3 = compute_jaccard(word_ngrams_3)

        if self.used_features['wn_dice_1'] == 1:
            wn_dice_1 = compute_dice(word_ngrams_1)
        if self.used_features['wn_dice_2'] == 1:
            wn_dice_2 = compute_dice(word_ngrams_2)
        if self.used_features['wn_dice_3'] == 1:
            wn_dice_3 = compute_dice(word_ngrams_3)

        if self.used_features['wn_overlap_1'] == 1:
            wn_overlap_1 = compute_overlap(word_ngrams_1)
        if self.used_features['wn_overlap_2'] == 1:
            wn_overlap_2 = compute_overlap(word_ngrams_2)
        if self.used_features['wn_overlap_3'] == 1:
            wn_overlap_3 = compute_overlap(word_ngrams_3)

        # check if any cn feature was used by the model before computing the character n_grams
        if any(key[:2] == 'cn' and value == 1 for key, value in self.used_features.items()):
            character_ngrams_2 = create_character_ngrams(preprocessed_corpus, 2)
            character_ngrams_3 = create_character_ngrams(preprocessed_corpus, 3)
            character_ngrams_4 = create_character_ngrams(preprocessed_corpus, 4)

        # compute distance coefficients for these ngrams
        if self.used_features['cn_jaccard_2'] == 1:
            cn_jaccard_2 = compute_jaccard(character_ngrams_2)
        if self.used_features['cn_jaccard_3'] == 1:
            cn_jaccard_3 = compute_jaccard(character_ngrams_3)
        if self.used_features['cn_jaccard_4'] == 1:
            cn_jaccard_4 = compute_jaccard(character_ngrams_4)

        if self.used_features['cn_dice_2'] == 1:
            cn_dice_2 = compute_dice(character_ngrams_2)
        if self.used_features['cn_dice_3'] == 1:
            cn_dice_3 = compute_dice(character_ngrams_3)
        if self.used_features['cn_dice_4'] == 1:
            cn_dice_4 = compute_dice(character_ngrams_4)

        if self.used_features['cn_overlap_2'] == 1:
            cn_overlap_2 = compute_overlap(character_ngrams_2)
        if self.used_features['cn_overlap_3'] == 1:
            cn_overlap_3 = compute_overlap(character_ngrams_3)
        if self.used_features['cn_overlap_4'] == 1:
            cn_overlap_4 = compute_overlap(character_ngrams_4)

        tmp_lexical_features = []

        for pair in range(len(corpus)//2):
            pair_features = []

            if self.used_features['wn_jaccard_1'] == 1:
                pair_features.append(wn_jaccard_1['jaccard'][pair])
            if self.used_features['wn_jaccard_2'] == 1:
                pair_features.append(wn_jaccard_2['jaccard'][pair])
            if self.used_features['wn_jaccard_3'] == 1:
                pair_features.append(wn_jaccard_3['jaccard'][pair])

            if self.used_features['wn_dice_1'] == 1:
                pair_features.append(wn_dice_1['dice'][pair])
            if self.used_features['wn_dice_2'] == 1:
                pair_features.append(wn_dice_2['dice'][pair])
            if self.used_features['wn_dice_3'] == 1:
                pair_features.append(wn_dice_3['dice'][pair])

            if self.used_features['wn_overlap_1'] == 1:
                pair_features.append(wn_overlap_1['overlap'][pair])
            if self.used_features['wn_overlap_2'] == 1:
                pair_features.append(wn_overlap_2['overlap'][pair])
            if self.used_features['wn_overlap_3'] == 1:
                pair_features.append(wn_overlap_3['overlap'][pair])

            if self.used_features['cn_jaccard_2'] == 1:
                pair_features.append(cn_jaccard_2['jaccard'][pair])
            if self.used_features['cn_jaccard_3'] == 1:
                pair_features.append(cn_jaccard_3['jaccard'][pair])
            if self.used_features['cn_jaccard_4'] == 1:
                pair_features.append(cn_jaccard_4['jaccard'][pair])

            if self.used_features['cn_dice_2'] == 1:
                pair_features.append(cn_dice_2['dice'][pair])
            if self.used_features['cn_dice_3'] == 1:
                pair_features.append(cn_dice_3['dice'][pair])
            if self.used_features['cn_dice_4'] == 1:
                pair_features.append(cn_dice_4['dice'][pair])

            if self.used_features['cn_overlap_2'] == 1:
                pair_features.append(cn_overlap_2['jaccard'][pair])
            if self.used_features['cn_overlap_3'] == 1:
                pair_features.append(cn_overlap_3['jaccard'][pair])
            if self.used_features['cn_overlap_4'] == 1:
                pair_features.append(cn_overlap_4['jaccard'][pair])

            pair_features_tuple = tuple(pair_features)
            tmp_lexical_features.append(pair_features_tuple)

        self.lexical_features = np.array(tmp_lexical_features)

        return self.lexical_features

    def extract_syntactic_features(self, corpus, pos_tags, dependencies):
        '''
        Parameters
        ----------


        '''

        if corpus is None:
            print("Argument corpus is empty, returning None")
            return

        if pos_tags:
            pipeline_tags = new_full_pipe(corpus, options={"pos_tagger":True, "string_or_array":True})
            tags = build_sentences_from_tokens(pipeline_tags.pos_tags)

            # compute POS tags
            pos_tags = compute_pos(tags)
            self.used_features.update({'pos_adj': 1, 'pos_adv': 1, 'pos_art': 1, 'pos_conj-c': 1, 'pos_conj-s': 1, 'pos_intj': 1, 'pos_n': 1, 'pos_n-adj': 1, 'pos_num': 1, 'pos_pron-det': 1, 'pos_pron-indp': 1, 'pos_pron-pers': 1, 'pos_prop': 1, 'pos_prp': 1, 'pos_punc': 1, 'pos_v-fin': 1, 'pos_v-ger': 1, 'pos_v-inf': 1, 'pos_v-pcp': 1})
        else:
            self.used_features.update({'pos_adj': 0, 'pos_adv': 0, 'pos_art': 0, 'pos_conj-c': 0, 'pos_conj-s': 0, 'pos_intj': 0, 'pos_n': 0, 'pos_n-adj': 0, 'pos_num': 0, 'pos_pron-det': 0, 'pos_pron-indp': 0, 'pos_pron-pers': 0, 'pos_prop': 0, 'pos_prp': 0, 'pos_punc': 0, 'pos_v-fin': 0, 'pos_v-ger': 0, 'pos_v-inf': 0, 'pos_v-pcp': 0})

        if dependencies:
            # compute Syntactic Dependency parsing
            dependencies = dependency_parsing(corpus)
            self.used_features.update({'dependency_parsing': 1})
        else:
            self.used_features.update({'dependency_parsing': 0})

        tmp_syntactic_features = []

        for pair in range(len(corpus)//2):
            pair_features = []

            if not pos_tags.empty:
                pair_features.append(pos_tags['adj'][pair])
                pair_features.append(pos_tags['adv'][pair])
                pair_features.append(pos_tags['art'][pair])
                pair_features.append(pos_tags['conj-c'][pair])
                pair_features.append(pos_tags['conj-s'][pair])
                pair_features.append(pos_tags['intj'][pair])
                pair_features.append(pos_tags['n'][pair])
                pair_features.append(pos_tags['n-adj'][pair])
                pair_features.append(pos_tags['num'][pair])
                pair_features.append(pos_tags['pron-det'][pair])
                pair_features.append(pos_tags['pron-indp'][pair])
                pair_features.append(pos_tags['pron-pers'][pair])
                pair_features.append(pos_tags['prop'][pair])
                pair_features.append(pos_tags['prp'][pair])
                pair_features.append(pos_tags['punc'][pair])
                pair_features.append(pos_tags['v-fin'][pair])
                pair_features.append(pos_tags['v-ger'][pair])
                pair_features.append(pos_tags['v-inf'][pair])
                pair_features.append(pos_tags['v-pcp'][pair])
            else:
                pair_features.extend([0]*19)

            if not dependencies.empty:
                pair_features.append(dependencies['dependency_parsing_jc'][pair])
            else:
                pair_features.extend([0]*1)

            pair_features_tuple = tuple(pair_features)
            tmp_syntactic_features.append(pair_features_tuple)

        self.syntactic_features = np.array(tmp_syntactic_features)

        return self.syntactic_features

    def _extract_syntactic_features(self, corpus):
        '''
        Parameters
        ----------


        '''

        if corpus is None:
            print("Argument corpus is empty, returning None")
            return

        # check if any pos feature was used by the model before computing the pipeline tags and respective features
        if any(key[:3] == 'pos' and value == 1 for key, value in self.used_features.items()):
            pipeline_tags = new_full_pipe(corpus, options={"pos_tagger":True, "string_or_array":True})
            tags = build_sentences_from_tokens(pipeline_tags.pos_tags)

            # compute POS tags
            pos_tags = compute_pos(tags)
        else:
            pos_tags = None

        if self.used_features['dependency_parsing'] == 1:
            # compute Syntactic Dependency parsing
            dependencies = dependency_parsing(corpus)
        else:
            dependencies = None

        if pos_tags is None and dependencies is None:
            return None
        else:
            tmp_syntactic_features = []

            for pair in range(len(corpus)//2):
                pair_features = []

                if pos_tags is not None:
                    if not pos_tags.empty:
                        if self.used_features['pos_adj'] == 1:
                            pair_features.append(pos_tags['adj'][pair])
                        if self.used_features['pos_adv'] == 1:
                            pair_features.append(pos_tags['adv'][pair])
                        if self.used_features['pos_art'] == 1:
                            pair_features.append(pos_tags['art'][pair])
                        if self.used_features['pos_conj-c'] == 1:
                            pair_features.append(pos_tags['conj-c'][pair])
                        if self.used_features['pos_conj-s'] == 1:
                            pair_features.append(pos_tags['conj-s'][pair])
                        if self.used_features['pos_intj'] == 1:
                            pair_features.append(pos_tags['intj'][pair])
                        if self.used_features['pos_n'] == 1:
                            pair_features.append(pos_tags['n'][pair])
                        if self.used_features['pos_n-adj'] == 1:
                            pair_features.append(pos_tags['n-adj'][pair])
                        if self.used_features['pos_num'] == 1:
                            pair_features.append(pos_tags['num'][pair])
                        if self.used_features['pos_pron-det'] == 1:
                            pair_features.append(pos_tags['pron-det'][pair])
                        if self.used_features['pos_pron-indp'] == 1:
                            pair_features.append(pos_tags['pron-indp'][pair])
                        if self.used_features['pos_pron-pers'] == 1:
                            pair_features.append(pos_tags['pron-pers'][pair])
                        if self.used_features['pos_prop'] == 1:
                            pair_features.append(pos_tags['prop'][pair])
                        if self.used_features['pos_prp'] == 1:
                            pair_features.append(pos_tags['prp'][pair])
                        if self.used_features['pos_punc'] == 1:
                            pair_features.append(pos_tags['punc'][pair])
                        if self.used_features['pos_v-fin'] == 1:
                            pair_features.append(pos_tags['v-fin'][pair])
                        if self.used_features['pos_v-ger'] == 1:
                            pair_features.append(pos_tags['v-ger'][pair])
                        if self.used_features['pos_v-inf'] == 1:
                            pair_features.append(pos_tags['v-inf'][pair])
                        if self.used_features['pos_v-pcp'] == 1:
                            pair_features.append(pos_tags['v-pcp'][pair])

                if dependencies is not None:
                    if not dependencies.empty:
                        if self.used_features['dependency_parsing'] == 1:
                            pair_features.append(dependencies['dependency_parsing_jc'][pair])

                pair_features_tuple = tuple(pair_features)
                tmp_syntactic_features.append(pair_features_tuple)

            self.syntactic_features = np.array(tmp_syntactic_features)

            return self.syntactic_features

    def extract_semantic_features(self, corpus, semantic_relations, ners):
        '''
        Parameters
        ----------


        '''

        if corpus is None:
            print("Argument corpus is empty, returning None")
            return

        if semantic_relations:
            pipeline_lemmas = new_full_pipe(corpus, options={"lemmatizer":True, "string_or_array":True})
            lemmas = build_sentences_from_tokens(pipeline_lemmas.lemas)
            # compute semantic relations coefficients
            semantic_relations = compute_semantic_relations(lemmas)
            self.used_features.update({'sr_antonyms': 1, 'sr_synonyms': 1, 'sr_hyperonyms': 1, 'sr_other': 1})
        else:
            self.used_features.update({'sr_antonyms': 0, 'sr_synonyms': 0, 'sr_hyperonyms': 0, 'sr_other': 0})

        if ners:
            pipeline_entities = new_full_pipe(corpus, options={"entity_recognition":True, "string_or_array":True})
            entities = build_sentences_from_tokens(pipeline_entities.entities)
            # compute NERs
            ners = compute_ner(entities)
            self.used_features.update({'all_ne': 1, 'ne_B-ABSTRACCAO': 1, 'ne_B-ACONTECIMENTO': 1, 'ne_B-COISA': 1, 'ne_B-LOCAL': 1, 'ne_B-OBRA': 1, 'ne_B-ORGANIZACAO': 1, 'ne_B-OUTRO': 1, 'ne_B-PESSOA': 1, 'ne_B-TEMPO': 1, 'ne_B-VALOR': 1})
        else:
            self.used_features.update({'all_ne': 0, 'ne_B-ABSTRACCAO': 0, 'ne_B-ACONTECIMENTO': 0, 'ne_B-COISA': 0, 'ne_B-LOCAL': 0, 'ne_B-OBRA': 0, 'ne_B-ORGANIZACAO': 0, 'ne_B-OUTRO': 0, 'ne_B-PESSOA': 0, 'ne_B-TEMPO': 0, 'ne_B-VALOR': 0})

        tmp_semantic_features = []

        for pair in range(len(corpus)//2):
            pair_features = []

            if not semantic_relations.empty:
                pair_features.append(semantic_relations['antonyms'][pair])
                pair_features.append(semantic_relations['synonyms'][pair])
                pair_features.append(semantic_relations['hyperonyms'][pair])
                pair_features.append(semantic_relations['other'][pair])
            else:
                pair_features.extend([0]*4)

            if not ners.empty:
                pair_features.append(ners['all_ners'][pair])
                pair_features.append(ners['B-ABSTRACCAO'][pair])
                pair_features.append(ners['B-ACONTECIMENTO'][pair])
                pair_features.append(ners['B-COISA'][pair])
                pair_features.append(ners['B-LOCAL'][pair])
                pair_features.append(ners['B-OBRA'][pair])
                pair_features.append(ners['B-ORGANIZACAO'][pair])
                pair_features.append(ners['B-OUTRO'][pair])
                pair_features.append(ners['B-PESSOA'][pair])
                pair_features.append(ners['B-TEMPO'][pair])
                pair_features.append(ners['B-VALOR'][pair])
            else:
                pair_features.extend([0]*11)

            pair_features_tuple = tuple(pair_features)
            tmp_semantic_features.append(pair_features_tuple)

        self.semantic_features = np.array(tmp_semantic_features)

        return self.semantic_features

    def _extract_semantic_features(self, corpus):
        '''
        Parameters
        ----------


        '''

        if corpus is None:
            print("Argument corpus is empty, returning None")
            return

        # check if any sr feature was used by the model before computing the pipeline lemmas and semantic relations features
        if any(key[:2] == 'sr' and value == 1 for key, value in self.used_features.items()):
            pipeline_lemmas = new_full_pipe(corpus, options={"lemmatizer":True, "string_or_array":True})
            lemmas = build_sentences_from_tokens(pipeline_lemmas.lemas)
            # compute semantic relations coefficients
            semantic_relations = compute_semantic_relations(lemmas)
        else:
            semantic_relations = None

        # check if any ne feature was used by the model before computing the pipeline entities and ners features
        if any(key[:2] == 'ne' and value == 1 for key, value in self.used_features.items()):
            pipeline_entities = new_full_pipe(corpus, options={"entity_recognition":True, "string_or_array":True})
            entities = build_sentences_from_tokens(pipeline_entities.entities)
            # compute NERs
            ners = compute_ner(entities)
        else:
            ners = None

        if semantic_relations is None and ners is None:
            return None
        else:
            tmp_semantic_features = []

            for pair in range(len(corpus)//2):
                pair_features = []

                if semantic_relations is not None:
                    if not semantic_relations.empty:
                        if self.used_features['sr_antonyms'] == 1:
                            pair_features.append(semantic_relations['antonyms'][pair])
                        if self.used_features['sr_synonyms'] == 1:
                            pair_features.append(semantic_relations['synonyms'][pair])
                        if self.used_features['sr_hyperonyms'] == 1:
                            pair_features.append(semantic_relations['hyperonyms'][pair])
                        if self.used_features['sr_other'] == 1:
                            pair_features.append(semantic_relations['other'][pair])

                if ners is not None:
                    if not ners.empty:
                        if self.used_features['all_ne'] == 1:
                            pair_features.append(ners['all_ners'][pair])
                        if self.used_features['ne_B-ABSTRACCAO'] == 1:
                            pair_features.append(ners['B-ABSTRACCAO'][pair])
                        if self.used_features['ne_B-ACONTECIMENTO'] == 1:
                            pair_features.append(ners['B-ACONTECIMENTO'][pair])
                        if self.used_features['ne_B-COISA'] == 1:
                            pair_features.append(ners['B-COISA'][pair])
                        if self.used_features['ne_B-LOCAL'] == 1:
                            pair_features.append(ners['B-LOCAL'][pair])
                        if self.used_features['ne_B-OBRA'] == 1:
                            pair_features.append(ners['B-OBRA'][pair])
                        if self.used_features['ne_B-ORGANIZACAO'] == 1:
                            pair_features.append(ners['B-ORGANIZACAO'][pair])
                        if self.used_features['ne_B-OUTRO'] == 1:
                            pair_features.append(ners['B-OUTRO'][pair])
                        if self.used_features['ne_B-PESSOA'] == 1:
                            pair_features.append(ners['B-PESSOA'][pair])
                        if self.used_features['ne_B-TEMPO'] == 1:
                            pair_features.append(ners['B-TEMPO'][pair])
                        if self.used_features['ne_B-VALOR'] == 1:
                            pair_features.append(ners['B-VALOR'][pair])

                pair_features_tuple = tuple(pair_features)
                tmp_semantic_features.append(pair_features_tuple)

            self.semantic_features = np.array(tmp_semantic_features)

            return self.semantic_features

    def extract_distributional_features(self, corpus, tfidf, word2vec_mdl=None, fasttext_mdl=None, ptlkb_mdl=None, glove_mdl=None, numberbatch_mdl=None):
        '''
        Parameters
        ----------


        '''

        if corpus is None:
            print("Argument corpus is empty, returning None")
            return

        if word2vec_mdl:
            word2vec = word2vec_model(word2vec_mdl, corpus, 0, 1, 0)

            word2vec_tfidf = word2vec_model(word2vec_mdl, corpus, 1, 1, 0)
            self.used_features.update({'word2vec': 1, 'word2vec_tfidf': 1})
        else:
            self.used_features.update({'word2vec': 0, 'word2vec_tfidf': 0})

        if fasttext_mdl:
            fasttext = fasttext_model(fasttext_mdl, corpus, 0, 1, 0)

            fasttext_tfidf = fasttext_model(fasttext_mdl, corpus, 1, 1, 0)
            self.used_features.update({'fasttext': 1, 'fasttext_tfidf': 1})
        else:
            self.used_features.update({'fasttext': 0, 'fasttext_tfidf': 0})

        if ptlkb_mdl:
            pipeline_lemmas = new_full_pipe(corpus, options={"lemmatizer":True, "string_or_array":True})
            lemmas = build_sentences_from_tokens(pipeline_lemmas.lemas)

            ptlkb = ptlkb_model(ptlkb_mdl, 0, 1, 0, lemmas)

            ptlkb_tfidf = ptlkb_model(ptlkb_mdl, 1, 1, 0, lemmas)
            self.used_features.update({'ptlkb': 1, 'ptlkb_tfidf': 1})
        else:
            self.used_features.update({'ptlkb': 0, 'ptlkb_tfidf': 0})

        if glove_mdl:
            glove = word_embeddings_model(glove_mdl, corpus, 0, 1, 0)

            glove_tfidf = word_embeddings_model(glove_mdl, corpus, 1, 1, 0)
            self.used_features.update({'glove': 1, 'glove_tfidf': 1})
        else:
            self.used_features.update({'glove': 0, 'glove_tfidf': 0})

        if numberbatch_mdl:
            numberbatch = word_embeddings_model(numberbatch_mdl, corpus, 0, 1, 0)

            numberbatch_tfidf = word_embeddings_model(numberbatch_mdl, corpus, 1, 1, 0)
            self.used_features.update({'numberbatch': 1, 'numberbatch_tfidf': 1})
        else:
            self.used_features.update({'numberbatch': 0, 'numberbatch_tfidf': 0})

        if tfidf:
            # compute tfidf matrix - padding was applied to vectors of different sizes by adding zeros to the smaller vector of the pair
            tfidf_corpus = preprocessing(corpus, 0, 0, 0, 1)
            tfidf_matrix = compute_tfidf_matrix(tfidf_corpus, 0, 0, 1)
            self.used_features.update({'tfidf': 1})
        else:
            self.used_features.update({'tfidf': 0})

        tmp_distributional_features = []

        for pair in range(len(corpus)//2):
            pair_features = []

            if word2vec_mdl:
                pair_features.append(word2vec[pair])
                pair_features.append(word2vec_tfidf[pair])
            else:
                pair_features.extend([0]*2)

            if fasttext_mdl:
                pair_features.append(fasttext[pair])
                pair_features.append(fasttext_tfidf[pair])
            else:
                pair_features.extend([0]*2)

            if ptlkb_mdl:
                pair_features.append(ptlkb[pair])
                pair_features.append(ptlkb_tfidf[pair])
            else:
                pair_features.extend([0]*2)

            if glove_mdl:
                pair_features.append(glove[pair])
                pair_features.append(glove_tfidf[pair])
            else:
                pair_features.extend([0]*2)

            if numberbatch_mdl:
                pair_features.append(numberbatch[pair])
                pair_features.append(numberbatch_tfidf[pair])
            else:
                pair_features.extend([0]*2)

            if tfidf:
                pair_features.append(tfidf_matrix[pair])
            else:
                pair_features.extend([0]*1)

            pair_features_tuple = tuple(pair_features)
            tmp_distributional_features.append(pair_features_tuple)

        self.distributional_features = np.array(tmp_distributional_features)

        return self.distributional_features

    def _extract_distributional_features(self, corpus, word2vec_mdl=None, fasttext_mdl=None, ptlkb_mdl=None, glove_mdl=None, numberbatch_mdl=None):
        '''
        Parameters
        ----------


        '''

        if corpus is None:
            print("Argument corpus is empty, returning None")
            return

        if word2vec_mdl:
            if self.used_features['word2vec'] == 1:
                word2vec = word2vec_model(word2vec_mdl, corpus, 0, 1, 0)
            if self.used_features['word2vec_tfidf'] == 1:
                word2vec_tfidf = word2vec_model(word2vec_mdl, corpus, 1, 1, 0)

        if fasttext_mdl:
            if self.used_features['fasttext'] == 1:
                fasttext = fasttext_model(fasttext_mdl, corpus, 0, 1, 0)
            if self.used_features['fasttext_tfidf'] == 1:
                fasttext_tfidf = fasttext_model(fasttext_mdl, corpus, 1, 1, 0)

        if ptlkb_mdl:
            if self.used_features['ptlkb'] == 1 or self.used_features['ptlkb_tfidf'] == 1:
                pipeline_lemmas = new_full_pipe(corpus, options={"lemmatizer":True, "string_or_array":True})
                lemmas = build_sentences_from_tokens(pipeline_lemmas.lemas)

                if self.used_features['ptlkb'] == 1:
                    ptlkb = ptlkb_model(ptlkb_mdl, 0, 1, 0, lemmas)

                if self.used_features['ptlkb_tfidf'] == 1:
                    ptlkb_tfidf = ptlkb_model(ptlkb_mdl, 1, 1, 0, lemmas)

        if glove_mdl:
            if self.used_features['glove'] == 1:
                glove = word_embeddings_model(glove_mdl, corpus, 0, 1, 0)

            if self.used_features['glove_tfidf'] == 1:
                glove_tfidf = word_embeddings_model(glove_mdl, corpus, 1, 1, 0)

        if numberbatch_mdl:
            if self.used_features['numberbatch'] == 1:
                numberbatch = word_embeddings_model(numberbatch_mdl, corpus, 0, 1, 0)

            if self.used_features['numberbatch_tfidf'] == 1:
                numberbatch_tfidf = word_embeddings_model(numberbatch_mdl, corpus, 1, 1, 0)

        if self.used_features['tfidf'] == 1:
            # compute tfidf matrix - padding was applied to vectors of different sizes by adding zeros to the smaller vector of the pair
            tfidf_corpus = preprocessing(corpus, 0, 0, 0, 1)
            tfidf_matrix = compute_tfidf_matrix(tfidf_corpus, 0, 0, 1)

        tmp_distributional_features = []

        for pair in range(len(corpus)//2):
            pair_features = []

            if word2vec_mdl:
                if self.used_features['word2vec'] == 1:
                    pair_features.append(word2vec[pair])
                if self.used_features['word2vec_tfidf'] == 1:
                    pair_features.append(word2vec_tfidf[pair])

            if fasttext_mdl:
                if self.used_features['fasttext'] == 1:
                    pair_features.append(fasttext[pair])
                if self.used_features['fasttext_tfidf'] == 1:
                    pair_features.append(fasttext_tfidf[pair])

            if ptlkb_mdl:
                if self.used_features['ptlkb'] == 1:
                    pair_features.append(ptlkb[pair])
                if self.used_features['ptlkb_tfidf'] == 1:
                    pair_features.append(ptlkb_tfidf[pair])

            if glove_mdl:
                if self.used_features['glove'] == 1:
                    pair_features.append(glove[pair])
                if self.used_features['glove_tfidf'] == 1:
                    pair_features.append(glove_tfidf[pair])

            if numberbatch_mdl:
                if self.used_features['numberbatch'] == 1:
                    pair_features.append(numberbatch[pair])
                if self.used_features['numberbatch_tfidf'] == 1:
                    pair_features.append(numberbatch_tfidf[pair])

            if self.used_features['tfidf'] == 1:
                pair_features.append(tfidf_matrix[pair])

            pair_features_tuple = tuple(pair_features)
            tmp_distributional_features.append(pair_features_tuple)

        self.distributional_features = np.array(tmp_distributional_features)

        return self.distributional_features

    def extract_multiple_features(self, corpus, to_store=1, extract_lexical=1, extract_syntactic=1, extract_semantic=1, extract_distributional=1, manual_feature_selection=None, word2vec_mdl=None, fasttext_mdl=None, ptlkb_mdl=None, glove_mdl=None, numberbatch_mdl=None):
        '''
        Parameters
        ----------


        '''
        if corpus is None:
            print("Argument corpus is empty, returning None")
            return

        # TODO: It would be awesome if we could just pass a dictionary with the features we want to extract
        if (extract_lexical and extract_syntactic and extract_semantic and extract_distributional) == 0:
            print("All feature groups are set to 0, so no features will be returned")
            return

        if manual_feature_selection is not None:
            self.feature_selection = 1
            self.used_features = self._update_used_features(manual_feature_selection)

        if extract_lexical:
            if self.lexical_features is None or to_store == 0:
                if self.feature_selection == 1:
                    lexical_features = self._extract_lexical_features(corpus)
                else:
                    lexical_features = self.extract_lexical_features(corpus, 1, 1, 1, 1, 1, 1)
            else:
                lexical_features = self.lexical_features
        else:
            lexical_features = None

        if extract_syntactic:
            if self.syntactic_features is None or to_store == 0:
                if self.feature_selection == 1:
                    syntactic_features = self._extract_syntactic_features(corpus)
                else:
                    syntactic_features = self.extract_syntactic_features(corpus, 1, 1)
            else:
                syntactic_features = self.syntactic_features
        else:
            syntactic_features = None

        if extract_semantic:
            if self.semantic_features is None or to_store == 0:
                if self.feature_selection == 1:
                    semantic_features = self._extract_semantic_features(corpus)
                else:
                    semantic_features = self.extract_semantic_features(corpus, 1, 1)
            else:
                semantic_features = self.semantic_features
        else:
            semantic_features = None

        if extract_distributional:
            if self.distributional_features is None or to_store == 0:
                if self.feature_selection == 1:
                    distributional_features = self._extract_distributional_features(corpus, word2vec_mdl, fasttext_mdl, ptlkb_mdl, glove_mdl, numberbatch_mdl)
                else:
                    distributional_features = self.extract_distributional_features(corpus, 1, word2vec_mdl, fasttext_mdl, ptlkb_mdl, glove_mdl, numberbatch_mdl)
            else:
                distributional_features = self.distributional_features
        else:
            distributional_features = None

        # before concatening all groups of features it is necessary to check if they were extracted or not
        all_features = np.concatenate([fg for fg in [lexical_features, syntactic_features, semantic_features, distributional_features] if fg is not None], axis=1)

        # Choose whether the extracted features should be stored in the model or just returned
        if to_store:
            self.all_features = all_features

            return self.all_features
        else:
            return all_features

    # TODO: Should the function return an instance of self, the predicted similarity or both?
    # TODO: Should it return the predicted similarity or only the predict_similarity function should be able to do that
    def run_model(self, regressor, train_features, train_target, use_feature_selection=0, eval_features=None, eval_target=None, test_features=None):
        '''
        Parameters
        ----------


        '''
        if use_feature_selection and self.feature_selection == 0:
            if eval_features is not None and eval_target is not None:
                f_selection_values = feature_selection(train_features, eval_features, train_target, eval_target, regressor, self.used_features)

                # If len(f_selection_values) is different than 3, it means that all feature selection techniques achieve worst performance than unsing all features
                if f_selection_values is not None:
                    selector = f_selection_values[0]
                    selected_train_features = f_selection_values[1]
                    self.used_features = f_selection_values[2]

                    self.model = regressor.fit(selected_train_features, train_target)
                else:
                    self.model = regressor.fit(train_features, train_target)

                self._update_number_features()

                if test_features is not None:
                    selected_features = selector.transform(test_features)
                    predicted_similarity = self.model.predict(selected_features)

                    return predicted_similarity
                else:
                    print("Missing test features. Provide them if you want to test the model.")
            else:
                self._update_number_features()

                print("Missing evaluation features/target necessary to perform feature selection.")
        else:
            self._update_number_features()

            self.model = regressor.fit(train_features, train_target)

            if test_features is not None:
                predicted_similarity = self.model.predict(test_features)

                return predicted_similarity
            else:
                print("Missing test features. Provide them if you want to test the model.")

    def predict_similarity(self, features):
        '''
        Parameters
        ----------


        '''
        # TODO: Does not work without the selector to transform the features in case feature selection was used
        if features is not None:
            predicted_similarity = self.model.predict(features)

            return predicted_similarity
        else:
            print("Features parameter is missing. Provide it in order to make a prediction.")

    # TODO: Verify if it makes sense to add a tests folder to each model in order to store tests performed by it
    def save_model(self):
        '''
        Parameters
        ----------


        '''
        # TODO: Add an option to save the ID of each pair of sentences if needed
        save_model_path = os.path.join(ROOT_PATH, "trained_models", self.model_name)

        if os.path.exists(save_model_path):
            update_model = str(input("Directory {} already exists. Do you want to update the existing model (Y/n)? ".format(save_model_path)))

            if update_model.lower() == "y":
                update_model = 1
            else:
                update_model = 0

        # TODO: When saving files, check if the content to be written is the same that already exists in the file and if so don't write it again. It could be done using an hash.
        if not os.path.exists(save_model_path) or update_model:

            if not os.path.exists(save_model_path):
                os.mkdir(save_model_path)

            if self.model is not None:
                sts_model_path = os.path.join(save_model_path, self.model_name)

                dump(self.model, sts_model_path)

            if self.lexical_features is not None:
                lexical_features_path = os.path.join(save_model_path, 'lexical_features.csv')

                np.savetxt(lexical_features_path, self.lexical_features, delimiter=",")

            if self.syntactic_features is not None:
                syntactic_features_path = os.path.join(save_model_path, 'syntactic_features.csv')

                np.savetxt(syntactic_features_path, self.syntactic_features, delimiter=",")

            if self.semantic_features is not None:
                semantic_features_path = os.path.join(save_model_path, 'semantic_features.csv')

                np.savetxt(semantic_features_path, self.semantic_features, delimiter=",")

            if self.distributional_features is not None:
                distributional_features_path = os.path.join(save_model_path, 'distributional_features.csv')

                np.savetxt(distributional_features_path, self.distributional_features, delimiter=",")

            if self.all_features is not None:
                all_features_path = os.path.join(save_model_path, 'all_features.csv')

                np.savetxt(all_features_path, self.all_features, delimiter=",")

            if self.used_features:
                used_features_path = os.path.join(save_model_path, 'used_features.txt')

                with open(used_features_path, 'w') as fp:
                    fp.write('{}\n'.format(self.model_name))
                    fp.write('{}\n'.format(self.number_features))

                    for key, value in self.used_features.items():
                        fp.write('{}: {}\n'.format(key, value))

                fp.close()

    def load_model(self, model_name):
        '''
        Parameters
        ----------


        '''
        dir_load_model_path = os.path.join(ROOT_PATH, 'trained_models', model_name)

        while not os.path.exists(dir_load_model_path):
            model_name = str(input("Directory {} does not exist. Insert a valid model name:".format(dir_load_model_path)))

            dir_load_model_path = os.path.join(ROOT_PATH, 'trained_models', model_name)

        self.model_name = model_name

        load_model_path = os.path.join(dir_load_model_path, model_name)
        self.model = load(load_model_path)

        used_features_path = os.path.join(dir_load_model_path, 'used_features.txt')
        if os.path.exists(used_features_path):
            self.feature_selection = 1

            with open(used_features_path) as ufp:
                for i, line in enumerate(ufp):
                    if i == 1:
                        self.number_features = int(line)
                    if i > 1:
                        split_line = line.split(': ')
                        self.used_features[split_line[0]] = int(split_line[1][:-1])

        lexical_features_path = os.path.join(dir_load_model_path, 'lexical_features.csv')
        if os.path.exists(lexical_features_path):
            self.lexical_features = np.loadtxt(lexical_features_path, delimiter=",")

        syntactic_features_path = os.path.join(dir_load_model_path, 'syntactic_features.csv')
        if os.path.exists(syntactic_features_path):
            self.syntactic_features = np.loadtxt(syntactic_features_path, delimiter=",")

        semantic_features_path = os.path.join(dir_load_model_path, 'semantic_features.csv')
        if os.path.exists(semantic_features_path):
            self.semantic_features = np.loadtxt(semantic_features_path, delimiter=",")

        distributional_features_path = os.path.join(dir_load_model_path, 'distributional_features.csv')
        if os.path.exists(distributional_features_path):
            self.distributional_features = np.loadtxt(distributional_features_path, delimiter=",")

        all_features_path = os.path.join(dir_load_model_path, 'all_features.csv')
        if os.path.exists(all_features_path):
            self.all_features = np.loadtxt(all_features_path, delimiter=",")

    def _update_number_features(self):
        '''
        Parameters
        ----------


        '''
        tmp_features = 0

        for value in self.used_features.values():
            if value == 1:
                tmp_features += 1

        self.number_features = tmp_features

    @staticmethod
    def _update_used_features(manual_features_selected):
        used_features = {'wn_jaccard_1': 1, 'wn_jaccard_2': 1, 'wn_jaccard_3': 1, 'wn_dice_1': 1, 'wn_dice_2': 1, 'wn_dice_3': 1, 'wn_overlap_1': 1, 'wn_overlap_2': 1, 'wn_overlap_3': 1, 'cn_jaccard_2': 1, 'cn_jaccard_3': 1, 'cn_jaccard_4': 1, 'cn_dice_2': 1, 'cn_dice_3': 1, 'cn_dice_4': 1, 'cn_overlap_2': 1, 'cn_overlap_3': 1, 'cn_overlap_4': 1, 'pos_adj': 1, 'pos_adv': 1, 'pos_art': 1, 'pos_conj-c': 1, 'pos_conj-s': 1, 'pos_intj': 1, 'pos_n': 1, 'pos_n-adj': 1, 'pos_num': 1, 'pos_pron-det': 1, 'pos_pron-indp': 1, 'pos_pron-pers': 1, 'pos_prop': 1, 'pos_prp': 1, 'pos_punc': 1, 'pos_v-fin': 1, 'pos_v-ger': 1, 'pos_v-inf': 1, 'pos_v-pcp': 1, 'dependency_parsing': 1, 'sr_antonyms': 1, 'sr_synonyms': 1, 'sr_hyperonyms': 1, 'sr_other': 1, 'all_ne': 1, 'ne_B-ABSTRACCAO': 1, 'ne_B-ACONTECIMENTO': 1, 'ne_B-COISA': 1, 'ne_B-LOCAL': 1, 'ne_B-OBRA': 1, 'ne_B-ORGANIZACAO': 1, 'ne_B-OUTRO': 1, 'ne_B-PESSOA': 1, 'ne_B-TEMPO': 1, 'ne_B-VALOR': 1, 'word2vec': 1, 'word2vec_tfidf': 1, 'fasttext': 1, 'fasttext_tfidf': 1, 'ptlkb': 1, 'ptlkb_tfidf': 1, 'glove': 1, 'glove_tfidf': 1, 'numberbatch': 1, 'numberbatch_tfidf': 1, 'tfidf': 1}

        for key, value in manual_features_selected.items():
            if key in used_features.keys():
                used_features[key] = value

        return used_features
