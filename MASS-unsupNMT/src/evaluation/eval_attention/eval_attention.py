import os
import io
import sys
import argparse
import torch

from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel, get_masks

from src.fp16 import network_to_half
from collections import OrderedDict
import seaborn
import pandas
import matplotlib.pyplot as plt
import logging
import matplotlib as mpl
mpl.use('Agg') # plot on server

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
    parser.add_argument("--output_path", type=str, default="", help="Output path")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
    
    # source text and target text
    parser.add_argument("--src_text", type=str, default="", help="Source language")
    parser.add_argument("--tgt_text", type=str, default="", help="Target language")
    
    # use debug mode    
    parser.add_argument("--debug", action="store_true", default=False)
    return parser

def draw_attention(attention_matrix, source_tokens, target_tokens, output_path):
    """ save attention heatmap to a given path
    Params:
        attention_matrix: 2d numpy array, matrix[i][j] means the attention weight of target_tokens[i] to source_tokens[j]
        source_tokens: list of source tokens
        target_tokens: list of target tokens
        output_path: path to save the heatmap
    """
    data_frame = pandas.DataFrame(attention_matrix, index=target_tokens, columns=source_tokens)
    seaborn.heatmap(
        data=data_frame,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=.5
    )
    plt.savefig(output_path)
    plt.close()

class MassAttentionEvaluator():
    def __init__(self, encoder, decoder, params, dico):
        self.encoder = encoder
        self.decoder = decoder
        self.params = params
        self.dico = dico
        self.debug = params.debug

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

        # decoding and get attention
        word_ids = [sent]
        x2, x2_lens, x2_langs =self.word_ids2batch(word_ids, tgt_lang)
        attention_outputs = self.decoder.get_cross_attention(x=x2, lengths=x2_lens, langs=x2_langs, causal=True, src_enc=encoded, src_len=x1_lens)
        
        return src_tokens, tgt_tokens, attention_outputs

    def eval_attention(self, src_sent, src_lang, tgt_lang, output_dir):
        
        src_tokens, tgt_tokens, attention_weights = self.translate_get_attention(src_sent, src_lang, tgt_lang)

        for layer_id in attention_weights.n_layers:
            for head_id in attention_weights.n_heads:
                output_path = os.path.join(output_dir, "layer-{}_head-{}.jpg".format(layer_id, head_id))
                draw_attention(attention_weights.get_attention(sentence_id=0, layer=layer_id, head=head_id), src_tokens, tgt_tokens, output_path)

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
    with open(params.src_path, 'r') as f:
        src_sent = f.readline().rstrip()

    evaluator.eval_attention(src_sent, params.src_lang, params.tgt_lang, params.dump_path)

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
