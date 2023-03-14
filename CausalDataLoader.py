import math
import os
import pickle
import re
import uuid
import xml.dom.minidom
from collections import Counter
from cProfile import label
from xml.dom.minidom import parse

import h5py
import numpy as np
import spacy
from keras.preprocessing.text import text_to_word_sequence
from nltk import LancasterStemmer, PorterStemmer, WordNetLemmatizer
from pynlp import StanfordCoreNLP


class CausalDataLoader:

    def __init__(self):
        """
        The dataloader is able to load data from a variety of sources (XML files, DB and HTTP)
        into CausalItems
        """
        self.base_dir = "/home/voku/vestra/"
        self.extvec_path = self.base_dir + "models/wiki_extvec"
        self.sql_conn = False
        self.http_conn = False
        self.lemma = StanfordCoreNLP(annotators="lemma")
        self.nlp = spacy.load("en_3")
        self.wordnet = WordNetLemmatizer()
        self.lancaster = LancasterStemmer()
        self.porter = PorterStemmer()

        # model constants
        self.MAX_WLEN = 58
        self.MAX_CLEN = 23
        self.EXTVEC_DIM = 300
        self.EXTVEC_N_SYMBOLS = 1476022

        # must be set when the training vocab size is known
        self.vocab_size = False

    def parse_xml(self, xml_file):
        """
        Takes a training/testing XML file and returns the training items as a list of CausalItems

        Parameters:
            - xml_file (str) :: The filepath to the xml file
        Returns:
            - causal_items (list) :: A list of causal items
        """
        causal_items = []
        tree = xml.dom.minidom.parse(xml_file)
        items = tree.documentElement.getElementsByTagName("item")
        for item in items:
            label_string = item.getAttribute("label")
            labels = self._unpack_labels(label_string)
            fragment = item.getElementsByTagName("sentence")
            fragment = fragment[0].childNodes[0].data
            causal_items.append(CausalItem(fragment, labels))
        return causal_items

    def tokenize_items(self, items):
        """
        Takes a list of items and returns them tokenized

        Parameters:
            - items (list[CausalItem]) :: A list of causal items, all of which should have `self.is_tokenized = False`

        Returns:
            - tokenized_items (list[CausalItem]) A list of causl items all of which are now tokenized 
        """
        remove_paren = re.compile('\((.*?)\)')
        tokenized_items = []

        if not items or not isinstance(items[0], CausalItem):
            raise Exception(
                "items list cannot be empty and must only contain CausalItems types")

        for item in items:
            if item.is_tokenized:
                # make sure the item isnt accidentally already tokenized
                continue
            item.sentence_fragment = remove_paren.sub(
                '', item.sentence_fragment)
            item.sentence_fragment = self._segment_word(item.sentence_fragment)
            item.is_tokenized = True
            item.cp_index = self._causal_phrase_index(item.sentence_fragment)

            # we need to remove the embedding tags here from the sentence because they are needed for the
            # item.cp_index
            item.sentence_fragment = [i for i in item.sentence_fragment if i not in [
                'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11']]
            item.char_fragments = [[list(w) for w in s]
                                   for s in item.sentence_fragment]

        return items

    def word_index(self, causal_items):
        """
        Takes the full unique vocabulary of words and returns the words_to_index and index_to_words mapping

        Returns:
            - word_to_index (dict)
            - index_to_word (dict)

        """
        counts = Counter()
        for item in causal_items:
            if not item.is_tokenized or not item.sentence_fragment:
                continue
            for fragment in item.sentence_fragment:
                counts.update(fragment)
        vocab = sorted(counts, key=counts.get, reverse=True)
        word_to_index = {w: i for i, w in enumerate(vocab, 1)}
        index_to_word = {i: w for w, i in word_to_index.items()}
        return word_to_index, index_to_word

    def char_index(self, causal_items):
        """
        Takes the full unique vocabulary of chars and returns the char_to_index and the index_to_char dicts for 
        mapping 

        Returns:
            - char_to_index (dict)
            - index_to_char (dict)
        """
        counts = Counter()
        for item in causal_items:
            if not item.is_tokenized or not item.char_fragment:
                continue
            for char in item.char_fragment:
                counts.update(sum(char, []))

        char_vocab = sorted(counts, key=counts.get, reverse=True)
        char_to_index = {w: i for i, w in enumerate(char_vocab, 1)}
        index_to_char = {i: w for w, i in char_to_index.items()}
        return char_to_index, index_to_char

    def set_vocab_size(self, word2index):
        """
        Sets the vocab size given the size of the word2idex dict
        """
        self.vocab_size = len(word2index)+1

    def gen_rand_extvec_embedding(self, seed=80085):
        """
        Generates a random embedding with the same scale as the extvec embedding

        Parameters:
            - seed (int) :: The seed to start the RNG

        Returns:
            - extvec_embedding (np.array) :: An array with dims (EXTVEC_N_SYMBOLS, EXTVEC_DIM)
        """
        np.random.seed(seed)
        scale = math.sqrt(3.0 / self.EXTVEC_DIM)
        embedding = np.random.uniform(
            low=-scale, high=scale, size=(self.vocab_size, self.EXTVEC_DIM))
        return embedding

    def extvec_weights_to_index(self, rand_extvec_embedding, extvec_index, extvec_embedding, index2word):
        """
        Extracts the embeddings in the extvec model to the index2word dict. This function modifies
        the values and passes them back to caller (except index2word which is just referenced)

        Parameters:
        - rand_extvec_embedding (np.array) :: The randomized extvec embedding
            - extvec_index (dict) :: The loaded extvec_index_dict
            - extvec_embedding (np.array) :: The embedding

        Returns:
            - rand_extvec_embedding, extvec_index, extvec_embedding
        """
        c = 0
        for i in range(1, self.vocab_size):
            w = index2word[i]
            g = extvec_index.get(w)
            if g is None:
                g = self.replace(w, extvec_index)
            if g is not None:
                rand_extvec_embedding[i, :] = extvec_embedding[g, :]
                c += 1
        return rand_extvec_embedding, extvec_index, extvec_embedding

    def load_extvec_model(self, extvec_path):
        """
        Loads the extvec model given the path

        Returns:
            - extvec_index_dict (dict) :: A mapping of the extvec corpus
            - extvec_emedding_weights (np.array) :: The embedding weights of the corpus
        """
        extvec_index_dict = {}
        extvec_embedding_weights = np.empty(
            (self.EXTVEC_N_SYMBOLS, self.EXTVEC_DIM))
        with open(extvec_path, encoding="utf-8") as f:
            i = 0
            for line in f:
                line = line.split(" ")
                word = line[0]
                extvec_index_dict[word] = i
                extvec_embedding_weights[i, :] = np.asarray(
                    line[1:], dtype="float32")
                i += 1
        return extvec_index_dict, extvec_embedding_weights

    def _segment_word(self, text_string):
        """
        Segemtns a sentence into its constituent words, punctation agnostic

        Parameters:
            - text_string (str) :: The input sentence or sentence_fragment to split

        Returns:
            - segments (list) :: The list of segmented words with their punctuation marks seperated
        """
        segments = text_to_word_sequence(
            text_string, filters='!#$&*+.%/<=>?@[\\]^_`{|}~\t\n', lower=False, split=' ')
        for s in [",", ":", ";", '"']:
            if s in [',', ':', ';', '"']:
                for i in range(len(segments)):
                    if segments[i].endswith(s) and segments[i] != s:
                        segments[i] = segments[i][:-1]
                        segments.insert(i+1, s)
                for i in range(len(segments)):
                    if segments[i].endswith(s) and segments[i] != s:
                        segments[i] = segments[i][:-1]
                        segments.insert(i+1, s)
            if s in ['"']:
                for i in range(len(segments)):
                    if segments[i].startswith(s) and segments[i] != s:
                        segments[i] = segments[i][1:]
                        segments.insert(i, s)
                for i in range(len(segments)):
                    if segments[i].startswith(s) and segments[i] != s:
                        segments[i] = segments[i][1:]
                        segments.insert(i, s)
        return segments

    def _unpack_labels(self, label_string):
        """
        Takes a label string and unpacks it into an array of entity labels

        Parameters:
            - label_string (str) :: The label string in the format "((ex,ey), (ex2,ey), ...)"
        Returns:
            - labels (list) :: A list of labels
        """
        if label_string == "Non-Causal":
            return []

        label_string = label_string[12:].split(",")
        label_string = [x.replace("(", "").replace(")", "")
                        for x in label_string]
        return [label_string[i:i+2] for i in range(0, len(label_string), 2)]

    def _causal_phrase_index(self, word_segments):
        """
        Takes a list of word segments and extracts the caual phrase index from the segments

        Parameters:
            - word_segments (list) :: The word segments

        Returns:
            cp_index (list) :: The causal phrase index
        """
        cp_index = []
        labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        for l in labels:
            e = []
            for i in range(len(word_segments)):
                if word_segments[i] == 'e'+l:
                    e.append(i-2*(int(l)-1))
                    for j in range(i+1, len(word_segments)):
                        if word_segments[j] == 'e'+l and j-2 != i:
                            e.append(j-2*int(l))
                    break
            if e != []:
                cp_index.append([i for i in range(e[0], e[-1]+1)])

        return cp_index

    def replace(self, w, d):
        """
        Replace the words that are not in the dictionary
        """
        r = d.get(w)
        if r is None:
            nw = w.lower()
            r = d.get(nw)
        if r is None:
            nw = [i[0].lemma for i in self.lemma(w)][0]
            r = d.get(nw)
        if r is None:
            nw = [i.lemma_ for i in self.nlp(w)][0]
            r = d.get(nw)
        if r is None:
            nw = self.wordnet.lemmatize(w)
            r = d.get(nw)
        if r is None:
            nw = self.porter.stem(w)
            r = d.get(nw)
        if r is None:
            nw = self.lancaster.stem(w)
            r = d.get(nw)
        if r is None:
            nw = w[:-1]
            r = d.get(nw)
        if r is None:
            nw = w[:-2]
            r = d.get(nw)
        return r

    def _dump_indices(self, save_dir, word_to_index, index_to_word, char_to_index, index_to_char):
        """
        Dumps a mapping, returning the directory name

        Parameters:
            - save_dir (str) :: The path to the master directory in which to save the temporary directory (no trailing slash)
        """
        if not os.path.isdir(save_dir):
            return False

        dirname = "{}/{}".format(save_dir, uuid.uuid4().hex)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        with open("{}/word_map.pkl".format(dirname), 'wb') as f:
            pickle.dump((word_to_index, index_to_word), f, -1)

        with open("{}/char_map.pkl".format(dirname), 'wb') as f:
            pickle.dump((char_to_index, index_to_char), f, -1)


class CausalItem:
    def __init__(self, sentence_fragment, labels=False, char_fragments=False, is_tokenized=False, tokenized_sentence=False, cp_index=False):
        """
        Wrapper Data class for every causal training item. Used for training and for converting
        article data --> causality model training data.

        Parameters:
            - sentence_fragment (str) :: The sentence fragment, completely tagged with its causal entities

            - label (list) :: The label list. If empty then the label is non-causal. Else each list contains
                              a sublist of entities with the first index being the cause and all later indices
                              being the effects.
                              Ex: `labels = [["e3", "e2"], ["e3", "e1"]]` implies entity 3 causes entity 1 and 2

            - char_fragments (False|list) :: If defined, this is a list of each of the chars of the sentence_fragment. Used during
                                             the training of the embedded step

            - is_tokenized (bool) :: If true, the input will be assumed to already be tokenized and parsed in the
                                     correct format as to facilitate training

            - tokenized_sentence(False|list) :: used to pass the tokenized sentence upon instantiation and when tokenizing

            - cp_index (False|list) :: The causality phrase index, used when assembling the dataset for training
        """
        self.sentence_fragment = sentence_fragment
        self.char_fragment = char_fragments
        self.tokenized_sentence = tokenized_sentence
        self.labels = [] if not labels else labels
        self.is_tokenized = is_tokenized
        self.cp_idx = False if not cp_index else cp_index

    def __repr__(self):
        return "{}\n{}\n{}\n\t{}".format(self.sentence_fragment, self.tokenized_sentence, self.labels, self.is_tokenized)


# TESTING
c = CausalDataLoader()
items = c.parse_xml("/home/voku/vestra/src/causallink/test-corpus.xml")
dataset = c.tokenize_items(items)

word_to_index, index_to_word = c.word_index(dataset)
char_to_index, index_to_char = c.char_index(dataset)

c._dump_indices("/home/voku", word_to_index, index_to_word,
                char_to_index, index_to_char)
print("Done")
