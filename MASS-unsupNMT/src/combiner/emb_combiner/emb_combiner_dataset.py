"""
Dataset for train and evaluate embedding combiner
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class EmbCombinerDataset(Dataset):

    def __init__(self, whole_word_path, splitter, dico):
        """

        Args:
            whole_word_path: str
                path to read whole word
            splitter: Splitter
                splitter object to split whole words
            dico:  Dictionary
                a mass dictionary object to index whole words and separated words
        """

        # get idx of each splitted words
        self.dico = dico  # for future use
        self.splitter = splitter
        self.whole_words_ids, self.splitted_words_ids = self.read_index_data(whole_word_path)

    def __getitem__(self, idx):
        sample = {
            "splitted_word_ids": self.splitted_words_ids[idx],
            "whole_word_id": self.whole_words_ids[idx]
        }
        return sample

    def read_index_data(self, whole_word_path):
        whole_word_ids = []
        splitted_word_ids = []
        with open(whole_word_path, 'r') as f:
            for line in f:
                word = line.rstrip()
                idx = self.dico.word2id[word]

                whole_word_ids.append(idx)

                subwords = self.splitter.split_word(word)
                subword_ids = [self.dico.word2id[subword] for subword in subwords]

                splitted_word_ids.append(subword_ids)

        return whole_word_ids, splitted_word_ids
