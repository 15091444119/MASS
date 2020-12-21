"""
For a mass vocab and mass training data,
we get vocab for each language, split into 10% 10% 80% train, dev, test data for training combiner
1. learn word2count from raw train src and raw train tgt
2. get vocab intersection in word2count and bped vocab
3. generate src word and tgt word (if a word count more in src than tgt, it is src, else tgt)
4. split train dev test
5. write into file
"""

import argparse
import random
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bped_vocab")
    parser.add_argument("--raw_train_src", help="source text not bped")
    parser.add_argument("--raw_train_tgt", help="source text not bped")
    parser.add_argument("--src_lang")
    parser.add_argument("--tgt_lang")
    parser.add_argument("--dest_dir")

    args = parser.parse_args()

    return args


def learn_word2count(file_path):
    count_dict = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip().split()
            for word in line:
                if word not in count_dict:
                    count_dict[word] = 1
                else:
                    count_dict[word] += 1

    return count_dict


def generate_src_tgt_word_vocab(bpe_vocab, src_word2count, tgt_word2count):
    """
    for each token in the bpe vocab,
    if its not a bpe subword, and length > 1 and in src_word2count and tgt_word2count
    we choose if it is a src word or a tgt word
    Args:
        bpe_vocab:
        src_word2count:
        tgt_word2count:

    Returns:
        tuple:
            src_vocab:
            tgt_vocab:
    """
    src_vocab = {}
    tgt_vocab = {}

    for token in bpe_vocab:

        # not a word
        if "@@" in token or len(token) <= 1:
            continue
        if token not in src_word2count and token not in tgt_word2count:
            continue

        # src or tgt
        word = token
        if word in src_word2count:
            if word in tgt_word2count:
                if src_word2count[word] > tgt_word2count[word]:
                    src_vocab[word] = src_word2count[word]
                else:
                    tgt_vocab[word] = tgt_word2count[word]
            else:
                src_vocab[word] = src_word2count[word]
        elif word in tgt_word2count:
            tgt_vocab[word] = tgt_word2count[word]

    return src_vocab, tgt_vocab


def write_vocab(vocab, path):
    with open(path, 'w') as f:
        for word, count in vocab.items():
            f.writelines("{} {}\n".format(word, count))


def write_train_dev_test(train, dev, test, prefix):
    write_vocab(train, prefix + ".train.txt")
    write_vocab(dev, prefix + ".dev.txt")
    write_vocab(test, prefix + ".test.txt")


def split_train_dev_test(vocab):
    keys = list(vocab.keys())
    random.shuffle(keys)

    dev_end = round(len(keys) * 0.1)
    dev_vocab = {key: vocab[key] for key in keys[:dev_end]}

    test_end = round(len(keys) * 0.2)
    test_vocab = {key: vocab[key] for key in keys[dev_end:test_end]}

    train_vocab = {key: vocab[key] for key in keys[test_end:]}

    return train_vocab, dev_vocab, test_vocab


def read_vocab(vocab_path):
    vocab = set()

    with open(vocab_path, 'r') as f:
        for line in f:
            token, _ = line.rstrip().split(' ')
            vocab.add(token)

    return vocab


def main(args):

    bpe_vocab = read_vocab(args.bped_vocab)

    src_word2count = learn_word2count(args.raw_train_src)
    tgt_word2count = learn_word2count(args.raw_train_tgt)

    src_word_vocab, tgt_word_vocab = generate_src_tgt_word_vocab(
        bpe_vocab=bpe_vocab,
        src_word2count=src_word2count,
        tgt_word2count=tgt_word2count
    )

    src_train, src_dev, src_test = split_train_dev_test(src_word_vocab)

    tgt_train, tgt_dev, tgt_test = split_train_dev_test(tgt_word_vocab)

    write_train_dev_test(
        train=src_train,
        dev=src_dev,
        test=src_test,
        prefix=os.path.join(args.dest_dir, args.src_lang)
    )

    write_train_dev_test(
        train=tgt_train,
        dev=tgt_dev,
        test=tgt_test,
        prefix=os.path.join(args.dest_dir, args.tgt_lang)
    )


if __name__ == "__main__":

    args = parse()
    main(args)
