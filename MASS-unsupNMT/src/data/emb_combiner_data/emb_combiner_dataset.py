"""
Dataset for train and evaluate embedding combiner
"""
import torch
import numpy as np
import pdb
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

    def __len__(self):
        return len(self.whole_words_ids)

    def __getitem__(self, idx):
        sample = {
            "splitted_word_ids": self.splitted_words_ids[idx],
            "whole_word_id": self.whole_words_ids[idx]
        }
        return sample

    def read_index_data(self, whole_word_path):
        whole_word_ids = []
        splitted_word_ids = []

        have_unk = 0
        with open(whole_word_path, 'r') as f:
            for line in f:
                word, _ = line.rstrip().split(' ')
                idx = self.dico.word2id[word]

                whole_word_ids.append(idx)

                subwords = self.splitter.split_word(word)
                subword_ids = []
                for subword in subwords:
                    if subword not in self.dico.word2id:
                        subword_ids.append(self.dico.unk_index)
                        have_unk += 1
                    else:
                        subword_ids.append(self.dico.word2id[subword])
                splitted_word_ids.append(subword_ids)

        print("{} unk".format(have_unk))

        return whole_word_ids, splitted_word_ids
