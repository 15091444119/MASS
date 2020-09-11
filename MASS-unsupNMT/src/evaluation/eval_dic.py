"""
Evaluate the quality of a dictionary given another ground truth dictionary
"""
import argparse
from .compare_bli import read_dict
from .bli import calculate_word_translation_accuracy


def eval_dic():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp_dict")
    parser.add_argument("ref_dict")
    parser.add_argument("--topk", default=1)
    params = parser.parse_args()

    hyp_dict = read_dict(params.hyp_dict)
    ref_dict = read_dict(params.ref_dict)

    acc = calculate_word_translation_accuracy(truth_dict=ref_dict, hyp_dict=hyp_dict)

    print("Dictionary accuracy: {acc}".format(acc=acc))
