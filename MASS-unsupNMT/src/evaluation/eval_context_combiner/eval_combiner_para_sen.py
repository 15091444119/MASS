"""
evaluate translation of splitted words

for a sentence pair [x, y], it's bped format is like [x', y']

if a word wx in x is assigned to wy, there are 4 types:

1. wx is separated by bpe but wy is not
2. wy is separated by bpe but wx is not
3. all are not separated
4. all are separated

whatever, we get the representation of wx and wy and calculate similarity between them
"""


def parser():





def main():
    """
    Returns:
        scores: dict containing 4 average distance
            scores["split_src_whole_tgt"]
            scores["split_src_split_tgt"]
            scores["whole_src_whole_tgt"]
            scores["whole_src_split_tgt"]
    """
    # read sentences
    # read bped sentences
    # read alignments
    # get masks
    # combiner encode
    # calculate scores
    # return scores