"""
For a mass vocab and mass training data,
we get vocab for each language, split into 10% 10% 80% train, dev, test data for training combiner
"""

import argparse
import random
import os

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab")
    parser.add_argument("--train_src")
    parser.add_argument("--train_tgt")
    parser.add_argument("--src_lang")
    parser.add_argument("--tgt_lang")
    parser.add_argument("--dest_dir")

    args = parser.parse_args()

    return args


def learn_token2count(vocab, file_path):
    count_dict = {}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip().split()
            for token in line:
                if token not in vocab:
                    continue
                if token in count_dict:
                    count_dict[token] = 1
                else:
                    count_dict[token] += 1

    return count_dict


def generate_src_tgt_vocab(vocab, src_token2count, tgt_token2count):
    src_vocab = {}
    tgt_vocab = {}

    for token in vocab:
        if token in src_token2count:
            if token in tgt_token2count:
                if src_token2count[token] > tgt_token2count[token]:
                    src_vocab[token] = src_token2count[token]
                else:
                    tgt_vocab[token] = tgt_token2count[token]
            else:
                src_vocab[token] = src_token2count[token]
        elif token in tgt_token2count:
            tgt_vocab[token] = tgt_token2count[token]

    return src_vocab, tgt_vocab


def write_vocab(vocab, path):
    with open(path, 'w') as f:
        for word, count in vocab.items():
            f.writelines("{} {}\n".format(word, vocab))


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
            token, _ = line.rstrip().split()
            vocab.add(token)
    return vocab


def main(args):

    vocab = read_vocab(args.vocab)

    src_token2count = learn_token2count(vocab, args.train_src)
    tgt_token2count = learn_token2count(vocab, args.train_tgt)

    src_vocab, tgt_vocab = generate_src_tgt_vocab(
        vocab=vocab,
        src_token2count=src_token2count,
        tgt_token2count=tgt_token2count
    )

    src_train, src_dev, src_test = split_train_dev_test(src_vocab)

    tgt_train, tgt_dev, tgt_test = split_train_dev_test(tgt_vocab)

    write_train_dev_test(
        train=src_train,
        dev=src_dev,
        test=src_test,
        prefix=os.path.join(args.dest_dir, args.src)
    )

    write_train_dev_test(
        train=tgt_train,
        dev=tgt_dev,
        test=tgt_test,
        prefix=os.path.join(args.dest_dir, args.tgt)
    )


if __name__ == "__main__":

    args = parse()
    main(args)

