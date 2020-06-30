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

class MassAttentionEvaluator():
    def __init__(self, encoder, decoder, params, dico):
        self.encoder = encoder
        self.decoder = decoder
        self.params = params
        self.dico = dico

    def get_batch(self, sent, lang):
        """
        Params:
            sent: list of strings
            lang: input language
        """
        word_ids = [torch.LongTensor([self.dico.index(w) for w in s.strip().split()])
                        for s in sent]
        return self.word_ids2batch(word_ids, lang)

    def word_ids2batch(self, word_ids, lang):
        """
        Params:
            word_ids: list of tensor containing word ids(no special tokens), refer to self.get_batch() for more things
            lang: input langauge
        """
        lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.params.pad_index)
        batch[0] = self.params.eos_index
        for j, s in enumerate(word_ids):
            if lengths[j] > 2:  # if sentence not empty
                batch[1:lengths[j] - 1, j].copy_(s)
            batch[lengths[j] - 1, j] = self.params.eos_index
        langs = batch.clone().fill_(self.params.lang2id[lang])
        return batch, lengths, langs

    def translate_get_attention(self, src_sent, src_lang, tgt_lang):
        """ translate source sentence
        Returns:
            target tokens
        """
        eos_token = self.dico.id2word[self.params.eos_index]
        params = self.params
        x1, x1_lens, x1_langs = self.get_batch([src_sent], src_lang)

        # src tokens
        src_tokens = [eos_token] + [self.dico.id2word[self.dico.index(w)] for w in src_sent.strip().split()] + [eos_token]

        # encode
        encoded = self.encoder('fwd', x=x1.cuda(), lengths=x1_lens.cuda(), langs=x1_langs.cuda(), causal=False)
        encoded = encoded.transpose(0, 1)

        # generate
        if params.beam == 1:
            decoded, dec_lengths = self.decoder.generate(encoded, x1_lens.cuda(), params.lang2id[tgt_lang], max_len=int(1.5 * x1_lens.max().item() + 10))
        else:
            decoded, dec_lengths = self.decoder.generate_beam(
                encoded, x1_lens.cuda(), params.lang2id[tgt_lang], beam_size=params.beam,
                length_penalty=params.length_penalty,
                early_stopping=False,
                max_len=int(1.5 * x1_lens.max().item() + 10))

        # remove delimiters
        sent = decoded[:, 0] # batch size is exactly one
        delimiters = (sent == params.eos_index).nonzero().view(-1)
        assert len(delimiters) >= 1 and delimiters[0].item() == 0
        sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]

        # target tokens
        tgt_tokens = [eos_token] + [self.dico.id2word[idx.item()] for idx in sent] + [eos_token]
        print(src_tokens, tgt_tokens)
        # decoding and get attention
        word_ids = [sent]
        x2, x2_lens, x2_langs =self.word_ids2batch(word_ids, tgt_lang)
        self_attention, cross_attention = self.decoder.get_attention(x=x2.cuda(), lengths=x2_lens.cuda(), langs=x2_langs.cuda(), causal=True, src_enc=encoded, src_len=x1_lens.cuda())
        
        return src_tokens, tgt_tokens, self_attention, cross_attention

    def eval_attention(self, src_sent, src_lang, tgt_lang, output_dir, method):
        """ evaluate all attentions
        Params:
            method: "all": all layer and all heads "heads_average": one figure for each layer "layer_average": one figure for each head "all_average": one figure for all weights
        """
        assert method in ["all", "heads_average", "layer_average", "all_average"]
        
        src_tokens, tgt_tokens, self_attention, cross_attention = self.translate_get_attention(src_sent, src_lang, tgt_lang)

        self_attention_output_prefix = os.path.join(output_dir, "self-attention")
        draw_multi_layer_multi_head_attention(tgt_tokens, tgt_tokens, self_attention, method, self_attention_output_prefix)

        cross_attention_output_prefix = os.path.join(output_dir, "cross-attention")
        draw_multi_layer_multi_head_attention(src_tokens, tgt_tokens, cross_attention, method, cross_attention_output_prefix)
        

def main(params):

    # initialize the experiment
    initialize_exp(params)

    # generate parser / parse parameters
    reloaded = torch.load(params.model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # update dictionary parameters
    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index', 'lang2id']:
        setattr(params, name, getattr(model_params, name))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=True).cuda().eval()
    decoder = TransformerModel(model_params, dico, is_encoder=False, with_output=True).cuda().eval()
    def package_module(modules):
        state_dict = OrderedDict()
        for k, v in modules.items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
        return state_dict
    encoder.load_state_dict(package_module(reloaded['encoder']))
    decoder.load_state_dict(package_module(reloaded['decoder']))
    encoder.eval()
    decoder.eval()
    
    evaluator = MassAttentionEvaluator(encoder, decoder, params, dico)
    with open(params.src_text, 'r') as f:
        src_sent = f.readline().rstrip()

    evaluator.eval_attention(src_sent, params.src_lang, params.tgt_lang, params.dump_path, "all_average")

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
