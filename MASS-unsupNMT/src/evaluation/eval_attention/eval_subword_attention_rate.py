"""
For a give sentence,
jkdhlkjevaluate if subword end is less attented than subword front
"""
import os
import pdb
import io
import sys
import argparse
import torch
import numpy as np

from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel, get_masks

from src.fp16 import network_to_half
from collections import OrderedDict
from .attention_drawer import draw_multi_layer_multi_head_attention
from src.evaluation.utils import load_mass_model, prepare_batch_input, word_ids2batch
import logging

logger = logging.getLogger()


def parse_params():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
    parser.add_argument("--src_path", type=str)

    # decoding
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=1)

    # draw attention

    params = parser.parse_args()

    return params


def translate_get_attention(
        encoder,
        decoder,
        dico,
        mass_params,
        src_path,
        src_lang,
        tgt_lang,
        beam,
        length_penalty):
    """ translate source sentence and get attention matrix
    Returns:
        target tokens
    """
    encoder.eval()
    decoder.eval()
    src_sent = []
    for line in open(src_path, 'r'):
        assert len(line.strip().split()) > 0
        src_sent.append(line)

    statistic = {"front": [], "end": [], "norm_rate": []}
    word_count_sum = 0
    bped_word_count_sum = 0

    for i in range(0, len(src_sent), 64):
        j = min(len(src_sent), i + 64)
        x1, x1_lens, x1_langs = prepare_batch_input(src_sent[i:j], src_lang, dico, mass_params)

        # encode
        encoded = encoder('fwd', x=x1.cuda(), lengths=x1_lens.cuda(), langs=x1_langs.cuda(), causal=False)
        encoded = encoded.transpose(0, 1)

        # generate
        if beam == 1:
            decoded, dec_lengths = decoder.generate(encoded, x1_lens.cuda(), mass_params.lang2id[tgt_lang],
                                                    max_len=int(1.5 * x1_lens.max().item() + 10))
        else:
            decoded, dec_lengths = decoder.generate_beam(
                encoded, x1_lens.cuda(), mass_params.lang2id[tgt_lang], beam_size=beam,
                length_penalty=length_penalty,
                early_stopping=False,
                max_len=int(1.5 * x1_lens.max().item() + 10))

        # remove delimiters
        word_ids = []
        for k in range(decoded.size(1)):
            sent = decoded[:, k]  # batch size is exactly one
            delimiters = (sent == mass_params.eos_index).nonzero().view(-1)
            assert len(delimiters) >= 1 and delimiters[0].item() == 0
            sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]
            # target tokens
            word_ids.append(sent)

        x2, x2_lens, x2_langs = word_ids2batch(word_ids, tgt_lang, mass_params)
        attention, cross_attention = decoder.get_attention(x=x2.cuda(), lengths=x2_lens.cuda(), langs=x2_langs.cuda(),
                                                           causal=True, src_enc=encoded, src_len=x1_lens.cuda())
        debug_src = [[dico.id2word[x.item()] for x in x1[:, idx]] for idx in range(j - i)]
        debug_tgt = [[dico.id2word[x.item()] for x in x2[:, idx]] for idx in range(j - i)]
        for sentence_id in range(x2_lens.size(0)):
            word_count, bped_word_count = calculate_word_and_bped_word(debug_tgt[sentence_id], x2_lens[sentence_id].item())
            word_count_sum += word_count
            bped_word_count_sum += bped_word_count

        for k in range(i, j):
            average_attention = cross_attention.single_layer_attention(k-i, 2)
            src_average_attention = average_attention.mean(dim=0)
            update_subword_front_subword_end(statistic, src_average_attention, x1[:, k-i], x1_lens[k-i].item(), dico, encoded[k-i])

    print(sum(statistic["front"]) / len(statistic["front"]))
    print(sum(statistic["end"]) / len(statistic["end"]))
    print(sum(statistic["norm_rate"]) / len(statistic["norm_rate"]))
    print(word_count_sum, bped_word_count_sum, bped_word_count_sum / word_count_sum)


def calculate_word_and_bped_word(sent, slen):
    sent = sent[1:slen - 1] # remove bos and eos
    idx = 0
    word_count = 0
    bped_word_count = 0
    while(idx < len(sent)):
        if "@@" in sent[idx]:
            while(idx < len(sent) and "@@" in sent[idx]):
                idx += 1
            if idx == len(sent):
                break
            word_count += 1
            bped_word_count += 1
        else:
            word_count += 1

        idx += 1

    return word_count, bped_word_count


def update_subword_front_subword_end(statistic, attention, x, slen, dico, encoded):

    assert slen == attention.size(0)

    def get_word(ii):
        idx = x[ii].item()
        return dico.id2word[idx]

    i = 0
    while(i < slen):
        word = get_word(i)
        if "@@" not in word:
            i = i + 1
            continue
        # find end
        front = i
        while(i < slen and "@@" in get_word(i)):
            i += 1
        if i != slen and i - front == 1:
            print(get_word(front), get_word(i))
            statistic["front"].append(attention[front].item())
            statistic["end"].append(attention[i].item())

            norm_att_front = attention[front].item() * torch.norm(encoded[front]).item()
            norm_att_end = attention[i].item() * torch.norm(encoded[i]).item()
            statistic["norm_rate"].append(norm_att_front / (norm_att_front + norm_att_end))
        i = i + 1



def main(params):
    # initialize the experiment
    initialize_exp(params)

    # generate parser / parse parameters
    dico, model_params, encoder, decoder = load_mass_model(params.model_path)

    translate_get_attention(
        encoder=encoder,
        decoder=decoder,dico=dico, mass_params=model_params, src_path=params.src_path, src_lang=params.src_lang, tgt_lang=params.tgt_lang,beam=params.beam,
        length_penalty=params.length_penalty
    )


if __name__ == '__main__':
    # generate parser / parse parameters
    params = parse_params()

    # check parameters
    assert os.path.isfile(params.model_path)
    assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang

    # translate
    with torch.no_grad():
        main(params)
