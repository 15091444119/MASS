"""
read source words and target translation, remove punctuation in translations, then
"""
import string
import pdb
import re
import argparse
from .compare_bli import read_dict
from .bli import calculate_word_translation_accuracy

def post_process(text):
    text = text.strip()
    text = re.sub(r'[{}]+'.format(string.punctuation + ' '),'',text)
    return text

def read(path):
    words = []
    with open(path, 'r') as f:
        for line in f:
            words.append(line.rstrip())
    return words

def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_words")
    parser.add_argument("hyp")
    parser.add_argument("ref_dict")
    args = parser.parse_args()


    ref_dict = read_dict(args.ref_dict)
    src_bped_words = read(args.src_words)
    tgt_hyp = read(args.hyp)
    tgt_hyp = [post_process(x) for x in tgt_hyp]
    assert len(src_bped_words) == len(tgt_hyp)

    dic = {src_bped_words[i].replace("@@", "").replace(" ", ""): [tgt_hyp[i]] for i in range(len(src_bped_words))}
    src_word2length = {tokens.replace("@@", "").replace(" ", ""): len(tokens.split()) for tokens in src_bped_words}
    whole_word_dic = {}
    seperated_word_dic = {}
    for src in dic:
        if src_word2length[src] == 1:
            whole_word_dic[src] = dic[src]
        else:
            seperated_word_dic[src] = dic[src]
    all_acc = calculate_word_translation_accuracy(truth_dict=ref_dict, translation=dic, topk=1)
    whole_acc = calculate_word_translation_accuracy(truth_dict=ref_dict, translation=whole_word_dic, topk=1)
    seperated_acc = calculate_word_translation_accuracy(truth_dict=ref_dict, translation=seperated_word_dic, topk=1)
    print("All: acc:{}, Whole:{}, Seperated:{}".format(all_acc, whole_acc, seperated_acc))

if __name__ == "__main__":
    eval()

