"""
1. length distribution of the separated word
2. training count distribution of the separated word
"""
from .separated_rate import read_sentences, group_tokens
import matplotlib.pyplot as plt
import argparse


def get_separated_word_length_distribution(sentences):
    length2num = {}

    for sentence in sentences:
        for word in sentence:
            word_length = len(word)
            if word_length > 1:
                if word_length in length2num:
                    length2num[word_length] += 1
                else:
                    length2num[word_length] = 1

    return length2num


def draw_separated_word_length_distribution(length2num, saved_path):
    x = sorted(length2num.keys())
    y = [length2num[key] for key in x]
    y = [value / sum(y) for value in y]

    plt.bar(x, y)
    plt.show()
    plt.savefig(saved_path)


def analyse_length_distribution(data_path, saved_path):
    sentences = read_sentences(data_path)
    sentences = group_tokens(sentences)

    length2num = get_separated_word_length_distribution(sentences)

    draw_separated_word_length_distribution(length2num, saved_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--saved_path")
    args = parser.parse_args()

    analyse_length_distribution(args.data_path, args.saved_path)


if __name__ == "__main__":
    main()
