# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#
# Translate sentences from the input stream.
# The model will be faster is sentences are sorted by length.
# Input sentences must have the same tokenization and BPE codes than the ones used in the model.
#
# Usage:
#     cat source_sentences.bpe | \
#     python translate.py --exp_name translate \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output
#

import os
import io
import sys
import argparse
import torch

from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp
from src.model.encoder import EncoderInputs
from src.model.seq2seq import DecodingParams
from src.model.context_combiner.context_combiner import load_combiner
from src.model.seq2seq.common_seq2seq import BaseSeq2Seq
from src.model.encoder.combiner_encoder import MultiCombinerEncoder
from src.model.encoder.common_encoder import CommonEncoder
from src.evaluation.utils import load_mass_model
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel

from src.fp16 import network_to_half


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--fp16", type=bool_flag, default=False, help="Run model with float16")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")

    # model / output paths
    parser.add_argument("--output_path", type=str, default="", help="Output path")

    parser.add_argument("--beam", type=int, default=1, help="Beam size")
    parser.add_argument("--length_penalty", type=float, default=1, help="length penalty")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    # for loading model
    parser.add_argument("--model_type", type=str, choices=["mass_only", "mass_with_two_combiners"])
    parser.add_argument("--mass", type=str, default="", required=True)
    if parser.parse_known_args()[0].model_type == "mass_with_two_combiners":
        parser.add_argument("--src_combiner", type=str, default="")
        parser.add_argument("--tgt_combiner", type=str, default="")

    return parser


def load_model(params):

    if params.model_type == "mass_only":
        dico, model_params, encoder, decoder = load_mass_model(params.mass)
        encoder = CommonEncoder(encoder, mask_index=model_params.mask_index)
        seq2seq_model = BaseSeq2Seq(encoder, decoder)
    elif params.model_type == "mass_with_two_combiners":
        dico, model_params, encoder, decoder = load_mass_model(params.mass)
        src_combiner = load_combiner(params.src_combiner)
        tgt_combiner = load_combiner(params.tgt_combiner)
        encoder = MultiCombinerEncoder(encoder, {params.src_id: src_combiner, params.tgt_id:tgt_combiner}, dico, model_params.mask_index)
        seq2seq_model = BaseSeq2Seq(encoder, decoder)
    else:
        raise ValueError

    return dico, model_params, seq2seq_model


def prepare_batch_input(dico, params, sents):
    word_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                for s in sents]
    lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
    batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(params.pad_index)
    batch[0] = params.eos_index
    for j, s in enumerate(word_ids):
        if lengths[j] > 2:  # if sentence not empty
            batch[1:lengths[j] - 1, j].copy_(s)
        batch[lengths[j] - 1, j] = params.eos_index
    langs = batch.clone().fill_(params.src_id)

    encoder_inputs = EncoderInputs(x1=batch.cuda(), len1=lengths.cuda(), lang_id=params.src_id, langs1=langs.cuda())
    tgt_lang_id = params.tgt_id
    decoding_params = DecodingParams(beam_size=params.beam, length_penalty=params.length_penalty, early_stopping=False)

    return encoder_inputs, tgt_lang_id, decoding_params


def main(params):

    # initialize the experiment
    logger = initialize_exp(params)

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    dico, model_params, seq2seq_model = load_model(params)
    seq2seq_model = seq2seq_model.cuda()
    seq2seq_model.eval()

    # set params
    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))

    params.src_id = model_params.lang2id[params.src_lang]
    params.tgt_id = model_params.lang2id[params.tgt_lang]

    # read sentences from stdin
    src_sent = []
    for line in sys.stdin.readlines():
        assert len(line.strip().split()) > 0
        src_sent.append(line)
    logger.info("Read %i sentences from stdin. Translating ..." % len(src_sent))

    f = io.open(params.output_path, 'w', encoding='utf-8')

    for i in range(0, len(src_sent), params.batch_size):

        cur_sents = src_sent[i: i + params.batch_size]

        # prepare batch
        encoder_inputs, tgt_lang_id, decoding_params = prepare_batch_input(dico, params, cur_sents)

        # encode source batch and translate it
        decoded, decoded_lengths = seq2seq_model.generate(encoder_inputs, tgt_lang_id, decoding_params)

        # convert sentences to words
        for j in range(decoded.size(1)):

            # remove delimiters
            sent = decoded[:, j]
            delimiters = (sent == params.eos_index).nonzero().view(-1)
            assert len(delimiters) >= 1 and delimiters[0].item() == 0
            sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]

            # output translation
            source = src_sent[i + j].strip()
            target = " ".join([dico[sent[k].item()] for k in range(len(sent))])
            sys.stderr.write("%i / %i: %s -> %s\n" % (i + j, len(src_sent), source, target))
            f.write(target + "\n")

    f.close()


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()


    # translate
    with torch.no_grad():
        main(params)
