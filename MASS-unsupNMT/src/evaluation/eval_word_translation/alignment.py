"""
    Align parallel data using fast-align, then get translation of each source word
"""
import argparse
import sys
import shutil
import subprocess
import random
import logging
import pdb
import os

WHOLEWORD="whole-word"
SEPERATEDWORD="seperated"

def filter_alignment_one2one(alignment):
    """ Delete one to many and many to one in the alignment
    Params:
        alignment: fastalign output, like "0-0 1-1 2-3"
    Returns:
        one-one alignment
    Example:
        alignment: "0-0 0-1 1-2 2-2 3-2 4-4"
        output: "4-4"
    """
    s2t = {}
    t2s = {}
    for word_align in alignment.rstrip().split(' '):
        src_id, tgt_id = word_align.split('-')
        if src_id not in s2t:
            s2t[src_id] = [tgt_id]
        else:
            s2t[src_id].append(tgt_id)
        if tgt_id not in t2s:
            t2s[tgt_id] = [src_id]
        else:
            t2s[tgt_id].append(src_id)

    filtered_alignment = []
    for src_id, tgt_id_list in s2t.items():
        if len(tgt_id_list) == 1:
            if len(t2s[tgt_id_list[0]]) == 1:
                filtered_alignment.append("{}-{}".format(src_id, tgt_id_list[0]))
    
    return ' '.join(filtered_alignment)

def calculate_whole_word_seperated_word_translation_acc(alignments, srcs, tgts, hyps, word_types):
    """
        Given the alignment, we choose one-one alignment to generate word-pairs,
        for each word, if it's translation is in hyp, we think this translate is good.
        We calculate an accuracy score based on the above process.
        So we can calculate accuarcy on bpe seperated words and none seperated words.
    Params:
        alignments: fast-align alignment result
        srcs: list of strings containing source sentences
        tgts: list of strings containing target sentences
        hyps: list of strings containing model hyp sentences
        word_types: list of list, each list containes types for words in source sentences, "whole-word" or "seperated"
    Returns:
        whole word translation accuarcy and seperated word translation accuarcy
    """
    whole_word_cnt = 0
    whole_word_correct_cnt = 0
    seperated_word_cnt = 0
    seperated_word_correct_cnt = 0

    assert len(alignments) == len(srcs) == len(tgts) == len(hyps) == len(word_types)

    for alignment, src, tgt, hyp, word_type in zip(alignments, srcs, tgts, hyps, word_types):
        one_one_alignment = filter_alignment_one2one(alignment)
        src = src.rstrip().split(' ')
        tgt = tgt.rstrip().split(' ')
        hyp = hyp.rstrip().split(' ')
        assert len(src) == len(word_type)

        for word_align in one_one_alignment.rstrip().split(' '):
            src_id, tgt_id = word_align.split('-')

            if word_type[src_id] == "whole-word":
                whole_word_cnt += 1
                if tgt[tgt_id] in hyp:
                    whole_word_correct_cnt += 1

            elif word_type[src_id] == "seperated":
                seperated_word_cnt += 1
                if tgt[tgt_id] in hyp:
                    whole_word_correct_cnt += 1
    
    whole_word_acc = whole_word_correct_cnt / whole_word_cnt
    seperated_word_acc = seperated_word_correct_cnt / seperated_word_cnt

    return whole_word_acc, seperated_word_acc

def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="source sentence path")
    parser.add_argument("--tgt", help="target sentence path")
    parser.add_argument("--bped_src", help="bped source sentence path")
    parser.add_argument("--hyp", help="hyp sentence path")
    parser.add_argument("--alignments", help="Alignment result path")
    params = parser.parse_args()

    return params

def read_alignments(alignment_path):
    """ Read alignments induced by fast-align """
    alignments = []
    with open(alignment_path, 'r') as f:
        for line in f:
            alignments.append(line.rstrip())
    return alignments

def read_sentences(path):
    """ Read sentences """
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            sentences.append(line.rstrip())
    return sentences

def extract_word_types(bped_sentences):
    """ 
    For each sentences, give the bped sentence, we can calculate if each raw word has been seperated using bpe
    Params:
        bped_sentenes: list of string of bped sentences(FastBpe)
    Returns:
        A list, each element is also a list, each element in the sublist list is a type string representing the word type of a word in 
        the raw sentence, if the word is seperated using bpe, SEPERATEDWORD, else WHOLEWORD
    """
    word_types = []
    for bped_sentence in bped_sentences:
        cur_sentence_word_type = []
        cur_word = []
        for token in bped_sentence.split(' '):
            cur_word.append(token)
            if not token.endswith("@@"): # meet end of a word
                if len(cur_word) == 1:
                    cur_sentence_word_type.append(WHOLEWORD)
                else:
                    cur_sentence_word_type.append(SEPERATEDWORD)
                cur_word = []
        word_types.append(cur_sentence_word_type)
    
    return word_types

def main(params):
    params = parse_params()
    alignments = read_alignments(params.alignments)
    srcs = read_sentences(params.src)
    tgts = read_sentences(params.tgt)
    hyps = read_sentences(params.hyps)
    bped_srcs = read_sentences(params.bped_src)
    ord_types = extract_word_types(srcs, bped_srcs)
    whole_word_acc, seperated_word_acc = calculate_whole_word_seperated_word_translation_acc(alignments, srcs, tgts, hyps, word_types)
    print("Whole word accuarcy:{} Seperated word accuarcy:{}".format(whole_word_acc, seperated_word_acc))



