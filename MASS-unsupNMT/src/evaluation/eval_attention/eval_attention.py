import os
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
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
    
    # source text and target text
    parser.add_argument("--src_text", type=str, default="", help="Source language")

    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--length_penalty", type=float, default=1)
    
    return parser

def translate_get_attention(
    encoder,
    decoder,
    dico,
    mass_params,
    src_sent,
    src_lang,
    tgt_lang,
    beam,
    length_penalty):
    """ translate source sentence and get attention matrix
    Returns:
        target tokens
    """
    eos_token = dico.id2word[mass_params.eos_index]
    x1, x1_lens, x1_langs = prepare_batch_input([src_sent], src_lang, dico, mass_params)

    # src tokens
    src_tokens = [eos_token] + [dico.id2word[dico.index(w)] for w in src_sent.strip().split()] + [eos_token]

    # encode
    encoded = encoder('fwd', x=x1.cuda(), lengths=x1_lens.cuda(), langs=x1_langs.cuda(), causal=False)
    encoded = encoded.transpose(0, 1)

    # generate
    if mass_params.beam == 1:
        decoded, dec_lengths = decoder.generate(encoded, x1_lens.cuda(), mass_params.lang2id[tgt_lang], max_len=int(1.5 * x1_lens.max().item() + 10))
    else:
        decoded, dec_lengths = decoder.generate_beam(
            encoded, x1_lens.cuda(), mass_params.lang2id[tgt_lang], beam_size=beam,
            length_penalty=length_penalty,
            early_stopping=False,
            max_len=int(1.5 * x1_lens.max().item() + 10))

    # remove delimiters
    sent = decoded[:, 0] # batch size is exactly one
    delimiters = (sent == mass_params.eos_index).nonzero().view(-1)
    assert len(delimiters) >= 1 and delimiters[0].item() == 0
    sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]

    # target tokens
    tgt_tokens = [eos_token] + [dico.id2word[idx.item()] for idx in sent] + [eos_token]
    print(src_tokens, tgt_tokens)
    # decoding and get attention
    word_ids = [sent]
    x2, x2_lens, x2_langs =word_ids2batch(word_ids, tgt_lang, mass_params)
    attention, cross_attention = decoder.get_attention(x=x2.cuda(), lengths=x2_lens.cuda(), langs=x2_langs.cuda(), causal=True, src_enc=encoded, src_len=x1_lens.cuda())
    
    return src_tokens, tgt_tokens, attention, cross_attention


def eval_attention(
    encoder,
    decoder,
    dico,
    mass_params,
    method,
    src_sent,
    src_lang,
    tgt_lang,
    output_dir,
    beam,
    length_penalty):
    """ evaluate all attentions
    Params:
        encoder, decoder, dico, mass_params: model elements loaded from load_mass_model
        method: method to draw attention
        src_sent: a string of source sentence, will be tokenized using .split()
        src_lang: source language
        tgt_sent:
        tgt_lang:
        output_dir:
        beam:
        length_penalty
    """
    assert method in ["all", "heads_average", "layer_average", "all_average"]
    
    encoder.eval()
    decoder.eval()

    src_tokens, tgt_tokens, self_attention, cross_attention = translate_get_attention(encoder, decoder, dico, mass_params, src_sent, src_lang, tgt_lang, beam, length_penalty)

    self_attention_output_prefix = os.path.join(output_dir, "self-attention")
    draw_multi_layer_multi_head_attention(tgt_tokens, tgt_tokens, self_attention, method, self_attention_output_prefix)

    cross_attention_output_prefix = os.path.join(output_dir, "cross-attention")
    draw_multi_layer_multi_head_attention(src_tokens, tgt_tokens, cross_attention, method, cross_attention_output_prefix)
        
def main(params):

    # initialize the experiment
    initialize_exp(params)

    # generate parser / parse parameters
    dico, model_params, encoder, decoder = load_mass_model(params.model_path) 
    
    with open(params.src_text, 'r') as f:
        src_sent = f.readline().rstrip()

    eval_attention(
        encoder=encoder,
        decoder=decoder,
        dico=dico,
        mass_params=model_params,
        method=params.method,
        src_sent=src_sent,
        src_lang=params.src_lang,
        tgt_lang=params.tgt_lang,
        output_dir=params.dump_path,
        beam=params.beam,
        length_penalty=params.length_penalty
    )

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    assert os.path.isfile(params.model_path)
    assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang

    # translate
    with torch.no_grad():
        main(params)
