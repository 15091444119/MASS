"""
Statistic on separate rate
"""

import argparse


def read_sentences(path):
    """ Read sentences """
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            sentences.append(line.rstrip())
    return sentences


def group_tokens(bped_sentences):
    """ group tokens of one word in to a list
    For example
    He@@ llo , na@@ ncy ! will be saved as [["he@@", "llo"], [","], ["na@@", "cy"], ["!"]]
    """

    sentences = []
    for bped_sentence in bped_sentences:
        cur_word = []
        cur_sentence = []
        for token in bped_sentence.split(' '):
            cur_word.append(token)
            if not token.endswith("@@"):
                cur_sentence.append(cur_word)
                cur_word = []
        if len(cur_word) != 0: # the end of sentence is a bpe token, like "你@@ 好@@"
            cur_sentence.append(cur_word)
        sentences.append(cur_sentence)
    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bped_path", type=str, required=True)
    args = parser.parse_args()

    grouped_bped_sentences = group_tokens(read_sentences(args.bped_path))

    num_words = 0
    num_separated_words = 0

    for grouped_bped_sentence in grouped_bped_sentences:
        num_words += len(grouped_bped_sentence)
        for word in grouped_bped_sentence:
            if len(word) > 1:
                num_separated_words += 1

    print("Separated rate: {}".format(num_separated_words / num_words))


if __name__ == "__main__":
    main()
