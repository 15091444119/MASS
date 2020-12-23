from logging import getLogger
from torch.utils.data import Dataset
import random
import pdb

logger = getLogger()


class BaseContextCombinerDataset(Dataset):

    def __init__(self, dico, splitter):
        """

        Args:
            labeled_data_path:
            dico:
            splitter:
        """
        self.dico = dico
        self.splitter = splitter


class WordSampleContextCombinerDataset(BaseContextCombinerDataset):

    def __init__(self, labeled_data_path, dico, splitter):
        super().__init__(dico=dico, splitter=splitter)
        data = read_index_labeled_data(labeled_data_path=labeled_data_path, dico=dico)    # all the training data

        self.splitted_word_id, self.data_grouped_by_word = group_word(data)

        self.data_id_iter = [0 for i in range(len(self.splitted_word_id))]

    def next_instance(self, train_id):
        """
        return the next training instance for the train word
        Args:
            train_id:

        Returns:

        """

        iter_id = self.data_id_iter[train_id]
        sentence = self.data_grouped_by_word[train_id][iter_id]
        iter_id += 1
        self.data_id_iter[train_id] = iter_id % len(self.data_grouped_by_word[train_id])

        return sentence

    def __len__(self):
        return len(self.splitted_word_id)

    def __getitem__(self, item):
        """
        Args:
            item:
        Returns:
            dict:
                original_sentence: list
                    index of the original sentence
                separated_sentence: list
                    index of the separated sentence
                mapper:   dict
                    map from original position to separated position
        """
        trained_word_dic_id = self.splitted_word_id[item]
        original_sentence = self.next_instance(item)

        mapper, splitted_sentence = split(
            split_word_id=trained_word_dic_id,
            sentence=original_sentence,
            dico=self.dico,
            splitter=self.splitter,
            consider_bos_eos_in_mapper=True
        )

        sample = {
            "original_sentence": original_sentence,
            "splitted_sentence": splitted_sentence,
            "mapper": mapper
        }

        return sample


class SentenceSampleContextCombinerDataset(BaseContextCombinerDataset):
    """
    dataset length is all the data
    """

    def __init__(self, dico, splitter, labeled_data_path):
        super().__init__(dico=dico, splitter=splitter)
        self.labeled_sentences = read_index_labeled_data(labeled_data_path=labeled_data_path, dico=self.dico)


    def __len__(self):
        return len(self.labeled_sentences)

    def __getitem__(self, item):
        splitted_word_id, original_sentence = self.labeled_sentences[item]

        mapper, splitted_sentence = split(
            split_word_id=splitted_word_id,
            sentence=original_sentence,
            dico=self.dico,
            splitter=self.splitter,
            consider_bos_eos_in_mapper=True
        )

        sample = {
            "original_sentence": original_sentence,
            "splitted_sentence": splitted_sentence,
            "mapper": mapper
        }

        return sample


def read_index_labeled_data(labeled_data_path, dico):
    data = []

    logger.info("Read from {}".format(labeled_data_path))
    with open(labeled_data_path, 'r') as f:
        for line_id, line in enumerate(f):
            if line_id % 500000 == 0:
                logger.info("{}".format(line_id))
            label, sentence = line.rstrip().split('\t', 1)
            assert label in dico.word2id
            label_id = dico.index(label)
            indexed_sentence = [dico.index(token) for token in sentence.split()]
            if len(indexed_sentence) > 300:
                print(len(indexed_sentence))
            data.append((label_id, indexed_sentence))
    logger.info("Done")

    return data


def split(split_word_id, sentence, dico, splitter, consider_bos_eos_in_mapper):
    """
    split the first word in the sentence

    Args:
        split_word_id:
        sentence:
        dico:
        splitter:
        consider_bos_eos_in_mapper:
            the sentnece don't have bos and eos, but we consider bos and eos in mapper
    Returns:

    """
    assert split_word_id in sentence
    position = sentence.index(split_word_id)

    splitted_word = splitter.split_word(dico.id2word[split_word_id])
    splitted_index = [dico.index(token) for token in splitted_word]

    splitted_sentence = sentence[:position] + splitted_index + sentence[position + 1:]

    mapper = {}

    for i in range(position):
        mapper[i] = (i, i + 1)

    mapper[position] = (position, position + len(splitted_index))

    for i in range(position + 1, len(sentence)):
        mapper[i] = (i + len(splitted_index) - 1, i + len(splitted_index))

    if consider_bos_eos_in_mapper:
        new_mapper = {}
        new_mapper[0] = (0, 1)  # bos

        # real words
        for i in range(len(sentence)):
            new_mapper[i + 1] = (mapper[i][0] + 1, mapper[i][1] + 1)

        new_mapper[len(sentence) + 1] = (len(splitted_sentence) + 1, len(splitted_sentence) + 2)  # eos
        mapper = new_mapper

    return mapper, splitted_sentence


def read_index_vocab(vocab_path, dico):
    train_id2dic_id = {}

    with open(vocab_path, 'r') as f:
        for line in f:
            token, count = line.rstrip().split(' ')
            idx = dico.word2id[token]
            train_id2dic_id[len(train_id2dic_id)] = idx

    return train_id2dic_id


def group_word(data):
    splitted_word_id2instances = {}

    for word_id, sentence in data:
        if word_id not in splitted_word_id2instances:
            splitted_word_id2instances[word_id] = [sentence]
        else:
            splitted_word_id2instances[word_id].append(sentence)

    word_ids = list(splitted_word_id2instances.keys())
    instances = list(splitted_word_id2instances[key] for key in word_ids)

    return word_ids, instances
