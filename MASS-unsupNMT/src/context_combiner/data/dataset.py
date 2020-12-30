from logging import getLogger
from torch.utils.data import Dataset
from src.data.data_utils import read_index_filter_data
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
        )

        sample = {
            "original_sentence": original_sentence,
            "splitted_sentence": splitted_sentence,
            "mapper": mapper
        }

        return sample


class MultiSplitContextCombinerDataset(BaseContextCombinerDataset):

    def __init__(self, dico, splitter, data_path, max_split_num=3):
        super().__init__(dico=dico, splitter=splitter)
        self.sentences = read_index_filter_data(data_path=data_path, dico=self.dico)
        self.max_split_num = max_split_num

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        original_sentence = self.sentences[item]
        mapper, splitted_sentence = multi_split(
            sentence=original_sentence,
            splitter=self.splitter,
            dico=self.dico,
            max_split_num=self.max_split_num
        )

        return {
            "original_sentence": original_sentence,
            "splitted_sentence": splitted_sentence,
            "mapper": mapper
        }


def multi_split(sentence, splitter, dico, max_split_num):
    """

    Args:
        sentence:  indexed sentence without special words
        splitter:
        dico:
        consider_bos_eos_in_mapper:
        max_split_num:

    Returns:

    """

    # get splittable tokens
    splittable_token_index = []
    for index, token_id in enumerate(sentence):
        if dico.is_special(token_id):
            continue
        token = dico.id2word[token_id]
        if "@@" not in token and len(token) >= 2:
            splittable_token_index.append(index)

    # choose tokens to be splitted
    n_split = random.randint(1, min(len(splittable_token_index), max_split_num))
    splitted_index = random.sample(splittable_token_index, k=n_split)

    # split and get new sentence
    splitted_sentence = []
    mapper = []
    for i in range(len(sentence)):
        if i in splitted_index:
            splitted_word = [dico.index(x) for x in splitter.split_word(dico.id2word[sentence[i]])]
            mapper[i] = (len(splitted_sentence), len(splitted_sentence) + len(splitted_word))
            splitted_sentence.extend(splitted_word)
        else:
            mapper[i] = (len(splitted_sentence), len(splitted_sentence) + 1)
            splitted_sentence.append(sentence[i])

    return mapper, splitted_sentence


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


def split(split_word_id, sentence, dico, splitter):
    """
    split the first word in the sentence

    Args:
        split_word_id:
        sentence:
        dico:
        splitter:
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
