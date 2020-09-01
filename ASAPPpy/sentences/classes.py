from ASAPPpy.scripts.tools import build_sentences_from_tokens

from NLPyPort.FullPipeline import new_full_pipe

class Sentence():
    def __init__(self, text, tags=None, lemmas=None, entities=None):
        self.text = text
        self.tags = tags
        self.lemmas = lemmas
        self.entities = entities

    def compute_tags(self):
        pipeline_tags = new_full_pipe([self.text], options={"pos_tagger":True, "string_or_array":True})
        tags = build_sentences_from_tokens(pipeline_tags.pos_tags)

        # The pipeline output is a list, so we access the first element to obtain the sentence.
        self.tags = tags[0]

    def compute_lemmas(self):
        pipeline_lemmas = new_full_pipe([self.text], options={"lemmatizer":True, "string_or_array":True})
        lemmas = build_sentences_from_tokens(pipeline_lemmas.lemas)

        # The pipeline output is a list, so we access the first element to obtain the sentence.
        self.lemmas = lemmas[0]

    def compute_entities(self):
        pipeline_entities = new_full_pipe([self.text], options={"entity_recognition":True, "string_or_array":True})
        entities = build_sentences_from_tokens(pipeline_entities.entities)

        # The pipeline output is a list, so we access the first element to obtain the sentence.
        self.entities = entities[0]

    def compute_all(self):
        pipeline_output = new_full_pipe([self.text], options={"string_or_array":True})

        tags = build_sentences_from_tokens(pipeline_output.pos_tags)
        lemmas = build_sentences_from_tokens(pipeline_output.lemas)
        entities = build_sentences_from_tokens(pipeline_output.entities)

        # The pipeline output is a list, so we access the first element to obtain the sentence.
        self.tags = tags[0]
        self.lemmas = lemmas[0]
        self.entities = entities[0]
        