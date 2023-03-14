from flair.embeddings import FlairEmbeddings, StackedEmbeddings, BertEmbeddings, ELMoEmbeddings
from flair.data import Sentence
import numpy as np
import re


class CausalEmbeddingLoader:

    def __init__(self):
        """
        The CausalEmbeddingLoader is responsible for taking data from the CausalDataLoader and 
        creating contextual string embeddings in Flair, BERT and ELMO, and finally constructing 
        the final dataset that can then be passed to the model for training.
        """
        self.MAX_WLEN = 58
        self.FLIAR_DIM = 4096
        self.BERT_DIM = 4096
        self.ELMO_DIM = 3072

        self.embedding_forward = FlairEmbeddings("news-forward")
        self.embedding_backward = FlairEmbeddings("news-backward")
        self.stacked_embedding = StackedEmbeddings(
            embeddings=[self.embedding_forward, self.embedding_backward])
        self.bert_embedding = BertEmbeddings("bert-large-cased")
        self.elmo_embedding = ELMoEmbeddings("original")

    def make_contextual_string_embeddings(self, sentences):
        """
        Takes a list of sentences and extracts the flair, bert and elmo embeddings from it

        Parameters:
            - sentences (list) :: The list of sentences

        Returns:
            - flair_res (list)
            - bert_res (list)
            - elmo_res (list)
        """
        flair_res = []
        bert_res = []
        elmo_res = []
        sentence_wrapper = [Sentence(' '.join(i)) for i in sentences]
        for sentence in sentence_wrapper:

            self.stacked_embedding.embed(sentence)
            flair_res.append(np.concatenate((np.array([np.array(
                token.embedding) for token in sentence]), np.zeros((self.MAX_WLEN-len(sentence), self.FLAIR_DIM))), axis=0))

            self.bert_embedding.embed(sentence)
            bert_res.append(np.concatenate((np.array([np.array(
                token.embedding) for token in sentence]), np.zeros((self.MAX_WLEN-len(sentence), self.BERT_DIM))), axis=0))

            self.elmo_embedding.embed(sentence)
            elmo_res.append(np.concatenate((np.array([np.array(
                token.embedding) for token in sentence]), np.zeros((self.MAX_WLEN-len(sentence), self.ELMO_DIM))), axis=0))

        return flair_res, bert_res, elmo_res

    def find_cp(self, label):
        """
        Extracts the cause and effect index from a label sentence

        Parameters:
            - label (str) :: The label

        Returns:
            - cause (str)
            - effect (str)
        """
        regex = '\((e.*?)\)'
        matches = re.findall(regex, label)
        matches = [i.split(',') for i in matches]
        cause = [i[0] for i in matches]
        effect = [i[-1] for i in matches]
        return cause, effect

    def tagging(self, s, n, w):
        """
        Tagging
        """
        for i in range(len(w)):
            if i == 0:
                s[w[i]] = n
            else:
                s[w[i]] = n+1
        return s
