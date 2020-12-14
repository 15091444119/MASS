from .separated_rate import read_sentences, group_tokens
import matplotlib.pyplot as plt
import argparse
import pdb


def read_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, 'r') as f:
        for line in f:
            token, count = line.rstrip().split(' ')
            vocab[token] = int(count)
    return vocab


def get_separated_word_train_count_distribution(grouped_sentences, vocab):
    distribution = []
    for sentence in grouped_sentences:
        for word in sentence:
            for token in word:
                if token in vocab:
                    distribution.append(vocab[token])


    return distribution


def draw_save_separated_word_train_count_distribution(distribution, saved_path):
    """

    Args:
        distribution: list of int

    Returns:

    """
    plt.hist(distribution, 100, range=(0, 10000))
    plt.show()
    plt.savefig(saved_path)


def analyse_count_distribution(data_path, vocab_path, saved_path):
    vocab = read_vocab(vocab_path)

    sentences = read_sentences(data_path)
    sentences = group_tokens(sentences)

    count_distribution = get_separated_word_train_count_distribution(sentences, vocab)
    draw_save_separated_word_train_count_distribution(count_distribution, saved_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--vocab_path")
    parser.add_argument("--saved_path")

    args = parser.parse_args()

    analyse_count_distribution(data_path=args.data_path, vocab_path=args.vocab_path, saved_path=args.saved_path)


if __name__ == "__main__":
    main()