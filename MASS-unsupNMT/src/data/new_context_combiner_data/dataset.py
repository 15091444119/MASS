from logging import getLogger
from torch.utils.data import Dataset
import random

logger = getLogger()


class BaseContextCombinerDataset(Dataset):

    def __init__(self, vocab_path, dico, splitter):
        """

        Args:
            vocab_path:
            labeled_data_path:
            dico:
            splitter:
        """
        self.dico = dico
        self.splitter = splitter
        self.train_id2dic_id = read_index_vocab(vocab_path=vocab_path, dico=dico)


class ContextCombinerTrainDataset(BaseContextCombinerDataset):

    def __init__(self, vocab_path, data_path, dico, splitter, max_instance=100000):
        """

        Args:
            vocab_path:
                trained vocabulary
            data_path:
                all the training sentences
            dico:
                dictionary
            splitter:
        """
        super().__init__(vocab_path=vocab_path, dico=dico, splitter=splitter)
        self.dic_id2train_id = {dic_id: train_id for train_id, dic_id in self.train_id2dic_id.items()}
        self.data = read_index_data(data_path=data_path, dico=dico)    # all the training data
        # shuffle data
        random.shuffle(self.data)
        self.train_id2data_ids = select_data_for_each_word(
            data=self.data,
            dic_id2train_id=self.dic_id2train_id,
            max_instance=max_instance
        )

        self.data_id_iter = [0 for i in range(len(self.train_id2data_ids))]

    def next_instance(self, train_id):
        """
        return the next training instance for the train word
        Args:
            train_id:

        Returns:

        """

        iter_id = self.data_id_iter[train_id]
        sentence = self.data[train_id][iter_id]
        iter_id += 1
        self.data_id_iter[train_id] = iter_id % len(self.train_id2data_ids[train_id])

        return sentence

    def __len__(self):
        return len(self.train_id2dic_id)

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

        trained_word_dic_id = self.train_id2dic_id[item]
        original_sentence = self.next_instance(item)

        mapper, splitted_sentence = split(
            split_word_id=trained_word_dic_id,
            sentence=original_sentence,
            dico=self.dico,
            splitter=self.splitter
        )

        sample = {
            "original_sentence": original_sentence,
            "splitted_sentence": splitted_sentence,
            "mapper": mapper
        }

        return sample


class ContextCombinerTestDataset(BaseContextCombinerDataset):

    def __init__(self, vocab_path, dico, splitter, labeled_dataset):
        super().__init__(vocab_path=vocab_path, dico=dico, splitter=splitter)
        self.labeled_sentences = read_index_labeled_data(labeled_dataset=labeled_dataset, vocab=set(self.train_id2dic_id.values()), dico=self.dico)

    def __len__(self):
        return len(self.labeled_sentences)

    def __getitem__(self, item):
        splitted_word_id, original_sentence = self.labeled_sentences[item]

        mapper, splitted_sentence = split(
            split_word_id=splitted_word_id,
            sentence=original_sentence,
            dico=self.dico,
            splitter=self.splitter
        )

        sample = {
            "original_sentence": original_sentence,
            "splitted_sentence": splitted_sentence,
            "mapper": mapper
        }

        return sample


def read_index_labeled_data(labeled_dataset, vocab, dico):
    data = []
    with open(labeled_dataset, 'r') as f:
        for line in f:
            label, sentence = line.rstrip().split('\t', 1)
            assert label in dico.word2id
            label_id = dico.index(label)
            assert label_id in vocab
            indexed_sentence = [dico.index(token) for token in sentence.split()]
            data.append((label_id, indexed_sentence))

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

    splitted_sentence = sentence[:position] + splitted_index + sentence[position:]

    mapper = {}

    for i in range(position):
        mapper[i] = (i, i + 1)

    mapper[position] = (position, position + len(splitted_index))

    for i in range(position, len(splitted_sentence)):
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


def read_index_data(data_path, dico):
    data = []

    empty_line = 0
    with open(data_path, 'r') as f:
        for line in f:
            line = line.rstrip().split(' ')

            if len(line) == 0:
                continue
            else:
                indexed_line = [dico.index(token) for token in line]
                data.append(indexed_line)

    logger.info("Empty sentence number:{}".format(empty_line))
    return data


def select_data_for_each_word(data, dic_id2train_id, max_instance):
    """

    Args:
        data:
        dic_id2train_id:
        max_instance: int
            max training instance for each word

    Returns:
        a dictionary, key is train_id, data is index of the training instances

    """
    train_id2data_ids = {}
    for sentence_id, sentence in enumerate(data):
        for dic_idx in sentence:
            train_id = dic_id2train_id[dic_idx]

            if train_id not in train_id2data_ids:
                train_id2data_ids[train_id] = [sentence_id]
            elif len(train_id2data_ids) < max_instance:
                train_id2data_ids[train_id].append(sentence_id)

    return train_id2data_ids
