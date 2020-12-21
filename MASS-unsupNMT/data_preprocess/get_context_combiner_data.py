"""
This scripts generate dev, test data for context combiner
For each word, we use 20 sentences for dev or test
So the code
1. read the dev/test whole word vocab
2. read the bped text
3. shuffle bped text
4. for each sentence, if it's suitable for any word, we use it as a dev/test sentence
    one sentence can be used for more than one time.
5. write into file in the format word\tsentence\n
"""

import argparse
from random import shuffle


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab")
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--max_instance", type=int, default=20)

    args = parser.parse_args()

    return args


def main(args):
    vocab = read_vocab(args.vocab)

    data = read_data(args.data)

    shuffle(data)

    word2sentences = generate_word2sentences(
        vocab=vocab,
        data=data,
        max_instance=args.max_instance
    )

    write_instances(word2sentences=word2sentences, output_path=args.output)


def read_vocab(vocab_path):
    vocab = set()
    with open(vocab_path, 'r') as f:
        for line in f:
            word, _ = line.rstrip().split(' ')
            vocab.add(word)
    return vocab


def read_data(data_path):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(line.rstrip())
    return data


def generate_word2sentences(vocab, data, max_instance):
    word2sentences = {}
    for sentence in data:
        for token in sentence.split():
            if token in vocab:
                if token in word2sentences:
                    if len(word2sentences[token]) < max_instance:
                        word2sentences[token].append(sentence)
                else:
                    word2sentences[token] = [sentence]
    return word2sentences


def write_instances(word2sentences, output_path):
    with open(output_path, 'w') as f:
        for word, sentences in word2sentences.items():
            for sentence in sentences:
                f.write("{}\t{}\n".format(word, sentence))


if __name__ == "__main__":
    args = parse_args()
    main(args)
