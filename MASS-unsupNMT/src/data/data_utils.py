import logging
import torch
import pdb

logger = logging.getLogger()


def read_index_filter_data(data_path, dico, untouchable_words):
    """
    read, index, filter unsplittable data
    Args:
        data_path:
        dico:
        untouchable_words: don't split these words

    Returns:
    """

    indexed_data = []
    discard = 0
    logger.info("Start read from {}".format(data_path))
    with open(data_path, 'r') as f:
        for line in f:
            line = line.rstrip().split()

            if len(line) == 0:
                discard += 1
                continue

            n_splittable_token = 0
            indexed_line = []
            for token in line:
                index = dico.index(token)
                if not dico.is_special(index):
                    if "@@" not in token and len(token) >= 2 and token not in untouchable_words:
                        n_splittable_token += 1
                indexed_line.append(index)

            if n_splittable_token >= 1:
                indexed_data.append(indexed_line)
            else:
                discard += 1

    logger.info("Read {} lines from {} discard {} unsplittable lines".format(len(indexed_data), data_path, discard))

    return indexed_data


def read_vocab(vocab_path):
    word2count = {}
    with open(vocab_path, 'r') as f:
        for line in f:
            word, count = line.rstrip().split(' ')
            word2count[word] = int(count)

    return word2count


def batch_sentences(sentences, pad_index, bos_index, eos_index, batch_first=False):
    """
    Args:
        sentences: list of sublist, each sublist contains several int
        pad_index:
        bos_index:
        eos_index:
        batch_first:
    Returns:
        tuple:
            batch_sentences: torch.LongTensor [slen, bs]
            lengths: torch.LongTensor [bs]
    """
    lengths = torch.LongTensor([len(s) + 2 for s in sentences])
    sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(pad_index)

    sent[0] = bos_index
    for i, s in enumerate(sentences):
        if lengths[i] > 2:  # if sentence not empty
            sent[1:lengths[i] - 1, i].copy_(torch.LongTensor(s))
        sent[lengths[i] - 1, i] = eos_index

    if batch_first:
        sent = sent.transpose(0, 1)

    return sent, lengths
