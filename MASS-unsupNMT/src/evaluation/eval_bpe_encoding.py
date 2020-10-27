"""
For a given sentence, we randomly split one whole word into bpe tokens to see the similarity between
the original encoded word and the word be splitted.
"""
import argparse
import torch
import random
import pdb

from src.combiner.splitter import ReduceOneBpeSplitter
from src.evaluation.utils import load_mass_model, encode_sentences


def split_word(sentence, train_vocab, splitter):
    """
    Randomly choose one word(with more than 2 tokens) and split it into 2 subword
    Params:
        sentence: list
            a list of words representing a sentence
        train_vocab: set
            word in the training corpus
        splitter: WholeWordSplitter

    Returns:
        splitted_sentence: list
            a list of words representing the splitted one
        origin_pos: int
            the position of the splited word in the original sentence
        splitted_pos: list
            the position of the splited word in the splitted sentence
    """

    # choose a position
    candidates = []
    for idx, word in enumerate(sentence):
        if len(word) >= 2 and word in train_vocab:
            candidates.append((idx, word))
    if len(candidates) == 0:
        raise ValueError
    choosed = random.choice(candidates)

    # get the splitted sentence
    origin_pos = choosed[0]
    splitted_sentence = []
    for idx, word in enumerate(sentence):
        if idx != origin_pos:
            splitted_sentence.append(word)
        else:
            splitted_word = splitter.split_word(word)
            start_position = len(splitted_sentence)
            splitted_sentence.extend(splitted_word)
            end_position = len(splitted_sentence)
            splitted_pos = list(range(start_position, end_position))
            print(splitted_word)

    if len(splitted_pos) != 2:
        pdb.set_trace()


    return splitted_sentence, origin_pos, splitted_pos


def read_corpus(path):
    """
    Params:
        path: str
    Returns:
        corpus: list
            each item is a list of words
    """
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            sentences.append(line.rstrip().split())
    return sentences


def get_mask(shape, batch_pos, de=0):
    """
    mask position in batch_pos
    """
    batch_size, max_length, dim = shape
    mask = torch.BoolTensor(batch_size, max_length).fill_(False).cuda()
    for idx, pos in enumerate(batch_pos):
        mask[idx][pos - de] = True
    mask = mask.unsqueeze(-1).expand(shape)

    return mask


def read_train_vocab(train_vocab_path):
    train_vocab = set()
    with open(train_vocab_path, 'r') as f:
        for line in f:
            word, count = line.rstrip().split()
            train_vocab.add(word)
    return train_vocab


class Statistics(object):

    def __init__(self):
        self.number = 0
        self.first_token_cos_sum = 0
        self.second_token_cos_sum = 0

    def add_statistics(self, first_sim, second_sim):
        assert len(first_sim) == len(second_sim)
        self.number += first_sim.size(0)
        self.first_token_cos_sum += first_sim.sum().item()
        self.second_token_cos_sum += second_sim.sum().item()

    @property
    def first_token_average_sim(self):
        return self.first_token_cos_sum / self.number

    @property
    def second_token_average_sim(self):
        return self.second_token_cos_sum / self.number


def main(params):
    splitter = ReduceOneBpeSplitter.from_code_path(args.codes_path)

    dico, model_params, encoder, _ = load_mass_model(args.model_path)

    sentences = read_corpus(params.corpus_path)

    train_vocab = read_train_vocab(params.train_vocab)

    statistic = Statistics()
    bad = 0
    for i in range(0, len(sentences), params.batch_size):
        j = min(i + params.batch_size, len(sentences))

        # prepare batch
        batch_origin_pos = []
        batch_splitted_pos = []
        origin_batch = []
        splitted_batch = []
        for k in range(i, j):
            original_sentence = sentences[k]
            try:
                splitted_sentence, origin_pos, splitted_pos = split_word(original_sentence, train_vocab, splitter)
            except ValueError:
                bad += 1
                continue
            assert len(splitted_pos) == 2
            batch_origin_pos.append(origin_pos)
            batch_splitted_pos.append(splitted_pos)
            splitted_batch.append(' '.join(splitted_sentence))
            origin_batch.append(' '.join(original_sentence))

        assert len(origin_batch) != 0

        # encode
        origin_encoded, origin_length = encode_sentences(encoder, dico, model_params, origin_batch, params.lang)
        splitted_encoded, splitted_length = encode_sentences(encoder, dico, model_params, splitted_batch, params.lang)

        # add one considering the bos token (This fix a bug of old version !!)
        first_token_pos = [poses[0] + 1 for poses in batch_splitted_pos]
        second_token_pos = [poses[1] + 1 for poses in batch_splitted_pos]
        batch_origin_pos = [pos + 1 for pos in batch_origin_pos]

        # get mask
        original_mask = get_mask(origin_encoded.size(), batch_origin_pos)
        first_token_mask = get_mask(splitted_encoded.size(), first_token_pos)
        second_token_mask = get_mask(splitted_encoded.size(), second_token_pos)

        # get representation
        batch_size = origin_encoded.size(0)
        original_word = origin_encoded.masked_select(original_mask).view(batch_size, -1)
        first_token = splitted_encoded.masked_select(first_token_mask).view(batch_size, -1)
        second_token = splitted_encoded.masked_select(second_token_mask).view(batch_size, -1)

        # calculating cos similarity
        first_token_sim = torch.nn.CosineSimilarity(dim=-1)(original_word, first_token)
        second_token_sim = torch.nn.CosineSimilarity(dim=-1)(original_word, second_token)


        # add to statistic
        statistic.add_statistics(first_token_sim, second_token_sim)

    print(bad)
    first_token_average_sim = statistic.first_token_average_sim
    second_token_average_sim = statistic.second_token_average_sim
    average = (first_token_average_sim + second_token_average_sim) / 2

    print("first token average sim: {}, second token average sim: {}, average: {}".format(first_token_average_sim, second_token_average_sim, average))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--codes_path", type=str, help="bpe codes", default="")
    parser.add_argument("--train_vocab")
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--batch_size", type=int)

    args = parser.parse_args()
    main(args)




