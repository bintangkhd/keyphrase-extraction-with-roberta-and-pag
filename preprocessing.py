import nltk
from nltk.corpus import stopwords
from nltk import CRFTagger, RegexpParser
from string import punctuation

class Preprocessing:
    def __init__(self, title, abstract):
        self.pos_tagger_model = "./pos_tagger_model/all_indo_man_tag_corpus_model.crf.tagger"
        self.doc_title = title
        self.doc_abstract = abstract
        self.data_input = ""
        self.token = []
        self.pos_tagged_token = []
        self.noun_phrase = []

    def combine_data(self):
        self.data_input = self.doc_title + ". " + self.doc_abstract
    
    def cleansing_data(self):
        data_tokens = nltk.word_tokenize(self.data_input.lower())
        special_chars = set(punctuation)
        
        temp = []
        temp = list(filter(lambda token: token not in temp and token not in special_chars, data_tokens))
        print("temp : ", temp)
        stop_words = set(stopwords.words("indonesian"))
        self.token = list(filter(lambda token: token not in stop_words, temp))


    def pos_tagging_token(self):
        ct = CRFTagger()
        ct.set_model_file(self.pos_tagger_model)
        self.pos_tagged_token = ct.tag_sents([self.token])

    def noun_phrase_chunking(self):
        GRAMMAR_FORMULA = r"""
            NP:
                {<JJ|CD><CD>} # Kata sifat yang diikuti cardinal number 
                {<JJ|CD><NN.*><CD>} # Kata sifat dan kata benda yang diikuti cardinal number
                {<NN.*|JJ|NNP|NND>{2}} # NN terdiri dari 2 Kata
                {<NN.*|JJ|NNP|NND>{1}} # NN terdiri dari 1 kata
                {<FW>{2}} # FW (foreign word) 2 kata
                {<FW.*>{1,3}} # FW (foreign word) maksimal 3 kata
                {<FW>{1}} # FW (foreign word) 1 kata
            """

        np_parser = RegexpParser(GRAMMAR_FORMULA)
        
        for sentence in self.pos_tagged_token:
            tagged_tokens = [(word, tag) for word, tag in sentence]
            tree = np_parser.parse(tagged_tokens)
            chunks = [] 

            for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
                leaves = subtree.leaves()
                np = " ".join(word for word, tag in leaves)
                chunks.append(np)


            self.noun_phrase.append(chunks)
        