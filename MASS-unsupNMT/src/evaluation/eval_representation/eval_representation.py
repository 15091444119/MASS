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
from .tsne import tsne, bilingual_tsne
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
    parser.add_argument("--output_path", type=str, default="", help="Output path")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
    
    # source text and target text
    parser.add_argument("--src_text", type=str, default="", help="Source language")
    parser.add_argument("--tgt_text", type=str, default="", help="Target language")
    

    return parser

def get_sen_representation(encoded, lengths, method):
    """ calculate sentence representation given encoded word representation 
    params:
        encoded: [bs, len, dim]
        lengths: [bs, len]
    Returns:
        torch.tensor of size: [bs, dim]
    """
    assert len(encoded.size()) == 3
    assert len(lengths.size()) == 1
    assert encoded.size(0) == lengths.size(0)

    if method == "max":
        mask, _ = get_masks(encoded.size(1), lengths, causal=False) # [bs, len]
        mask = (mask == 0).unsqueeze(-1).expand_as(encoded)
        if torch.cuda.is_available():
            mask = mask.cuda()

        # set masked positions to -inf
        encoded = encoded.masked_fill(mask, -float('inf'))
        encoded, _ = torch.max(encoded, dim=1) # max pool along length
        return encoded

    elif method == "avg":
        mask, _ = get_masks(encoded.size(1), lengths, causal=False) # [bs, len]
        if torch.cuda.is_available():
            mask = mask.cuda()

        # set masked positions to zero
        encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)

        encoded = torch.sum(encoded, dim=1) # [bs, dim]
        encoded = encoded / lengths.unsqueeze(-1).to(encoded.dtype)
        return encoded
        
    elif method == "cls":
        return encoded[:,0,:]

def representations_cos_average(src_representation, tgt_representation):
    """ Average cos similarity of source sentences and target sentences """
    assert src_representation.size(0) == tgt_representation.size(0)
    logger.info("{}".format(src_representation.size(0)))
    #tgt_representation_reverse =torch.clone(tgt_representation)
    #for i in range(tgt_representation.size(0)):
    #    tgt_representation_reverse[i] = tgt_representation[tgt_representation.size(0) - 1 - i]
    cos_simi = torch.nn.functional.cosine_similarity(src_representation, tgt_representation, dim=-1)
    return cos_simi.mean().item()


class MassRepEvaluator():
    def __init__(self, encoder, decoder, params, dico, sen_rep_method):
        self.encoder = encoder
        self.decoder = decoder
        self.params = params
        self.dico = dico
        self.sen_rep_method = sen_rep_method
    
    def read_data(self, path):
        sent = []
        with open(path, 'r') as f:
            for line in f:
                assert len(line.strip().split()) > 0
                sent.append(line)
            logger.info("Read %{} sentences from {}. ".format(len(sent), path))
        return sent

    def get_batch(self, sent, lang):
        word_ids = [torch.LongTensor([self.dico.index(w) for w in s.strip().split()])
                            for s in sent]
        lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.params.pad_index)
        batch[0] = self.params.eos_index
        for j, s in enumerate(word_ids):
            if lengths[j] > 2:  # if sentence not empty
                batch[1:lengths[j] - 1, j].copy_(s)
            batch[lengths[j] - 1, j] = self.params.eos_index
        langs = batch.clone().fill_(self.params.lang2id[lang])
        return batch, lengths, langs

    def add_batch_sen_rep(self, cur_encoded_batch, all_layer_sen_rep, lengths):
        """ extract sentence rep from word rep, and append to all_layer_sen_rep """
        for layer_id, layer_encoded in enumerate(cur_encoded_batch):
            layer_sen_rep = get_sen_representation(layer_encoded, lengths.cuda(), self.sen_rep_method)
            all_layer_sen_rep[layer_id].append(layer_sen_rep)

    def cat_batches(self, all_layer_rep):
        """ cat tensor in one layer (each tensor contains representation of one batch) """
        for layer_id, layer_rep in enumerate(all_layer_rep):
            all_layer_rep[layer_id] = torch.cat(layer_rep, dim=0)

    def encode_src(self, sent, lang):
        """ encode sentences """
        all_layer_sen_rep = [[] for i in range(self.encoder.n_layers + 1)] 

        for i in range(0, len(sent), self.params.batch_size):
            x, x_lengths, x_langs = self.get_batch(sent[i:i + self.params.batch_size], lang)

            encoded, _ = self.encoder.get_all_layer_outputs(x=x.cuda(), lengths=x_lengths.cuda(), langs=x_langs.cuda(), causal=False)

            self.add_batch_sen_rep(encoded, all_layer_sen_rep, x_lengths.cuda())
            
        self.cat_batches(all_layer_sen_rep)
        return all_layer_sen_rep

    def encode_src_tgt(self, src_sent, tgt_sent, src_lang, tgt_lang):
        """ encoder + decoder to encode parallel data """
        assert len(src_sent) == len(tgt_sent)

        encoder_all_layer_sen_rep = [[] for i in range(self.encoder.n_layers + 1)]
        decoder_all_layer_sen_rep = [[] for i in range(self.decoder.n_layers + 1)]

        for i in range(0, len(src_sent), self.params.batch_size):
            x1, x1_lengths, x1_langs = self.get_batch(src_sent[i:i + params.batch_size], src_lang)
            x2, x2_lengths, x2_langs = self.get_batch(tgt_sent[i:i + params.batch_size], tgt_lang)

            # encode
            encoded, last_layer = self.encoder.get_all_layer_outputs(x=x1.cuda(), lengths=x1_lengths.cuda(), langs=x1_langs.cuda(), causal=False)
            self.add_batch_sen_rep(encoded, encoder_all_layer_sen_rep, x1_lengths.cuda())

            # decode
            last_layer = last_layer.transpose(0, 1)
            decoded, _ = self.decoder.get_all_layer_outputs(x=x2.cuda(), lengths=x2_lengths.cuda(), langs=x2_langs.cuda(), causal=True, src_enc=last_layer, src_len=x1_lengths.cuda())
            self.add_batch_sen_rep(decoded, decoder_all_layer_sen_rep, x2_lengths.cuda())
        
        self.cat_batches(encoder_all_layer_sen_rep)
        self.cat_batches(decoder_all_layer_sen_rep)

        return encoder_all_layer_sen_rep, decoder_all_layer_sen_rep
        
    def eval_encoder(self, src_lang, tgt_lang, src_path, tgt_path, tsne_saved_dir):
        logger.info("Eval encoder: Start...")

        src_sent = self.read_data(src_path)
        tgt_sent = self.read_data(tgt_path)

        src_all_layer_rep = self.encode_src(src_sent, src_lang)
        tgt_all_layer_rep = self.encode_src(tgt_sent, tgt_lang)

        for layer_id, (src_rep, tgt_rep) in enumerate(zip(src_all_layer_rep, tgt_all_layer_rep)):
            # cos similarity
            cos = representations_cos_average(src_rep, tgt_rep)
            logger.info("Encoder layer:{} Cos:{}".format(layer_id, cos))

            # tsne
            cated_rep = torch.cat([src_rep, tgt_rep], dim=0).cpu().numpy()
            labels = [0 for i in range(len(src_sent))] + [1 for i in range(len(tgt_sent))]
            tsne(cated_rep, labels, os.path.join(tsne_saved_dir, "eval_encoder_layer{}".format(layer_id)))
    
        logger.info("Eval encoder: Done.")

    def eval_encoder_decoder(self, src_lang, tgt_lang, src_path, tgt_path, tsne_saved_dir):
        logger.info("Eval encoder decoder: Start...")

        src_sent = self.read_data(src_path)
        tgt_sent = self.read_data(tgt_path)

        s2t_encoder_rep, s2t_decoder_rep = self.encode_src_tgt(src_sent, tgt_sent, src_lang, tgt_lang)
        s2s_encoder_rep, s2s_decoder_rep = self.encode_src_tgt(src_sent, src_sent, src_lang, src_lang)
        t2s_encoder_rep, t2s_decoder_rep = self.encode_src_tgt(tgt_sent, src_sent, tgt_lang, src_lang)
        t2t_encoder_rep, t2t_decoder_rep = self.encode_src_tgt(tgt_sent, tgt_sent, tgt_lang, tgt_lang)

        # encoder
        for layer_id, (s2t_rep, s2s_rep, t2s_rep, t2t_rep) in enumerate(zip(s2t_encoder_rep, s2s_encoder_rep, t2s_encoder_rep, t2t_encoder_rep)):
            # cos similarity compared with s2s
            cos_s2t_s2s = representations_cos_average(s2t_rep, s2s_rep)
            cos_t2s_s2s = representations_cos_average(t2s_rep, s2s_rep)
            cos_t2t_s2s = representations_cos_average(t2t_rep, s2s_rep)
            logger.info("Encoder layer:{} Cos s2t-s2s:{} Cos t2s-s2s:{} Cos t2t-s2s:{}".format(layer_id, cos_s2t_s2s, cos_t2s_s2s, cos_t2t_s2s))

            # draw tsne
            cated_rep = torch.cat([s2t_rep, s2s_rep, t2s_rep, t2t_rep], dim=0).cpu().numpy()
            labels = [0 for i in range(len(src_sent))] + [1 for i in range(len(src_sent))] + \
                [2 for i in range(len(src_sent))] + [3 for i in range(len(src_sent))]
            tsne(cated_rep, labels, os.path.join(tsne_saved_dir, "eval_encoder_decoder_encoder_layer{}".format(layer_id)))

        # decoder
        for layer_id, (s2t_rep, s2s_rep, t2s_rep, t2t_rep) in enumerate(zip(s2t_decoder_rep, s2s_decoder_rep, t2s_decoder_rep, t2t_decoder_rep)):
            # cos similarity compared with s2s
            cos_s2t_s2s = representations_cos_average(s2t_rep, s2s_rep)
            cos_t2s_s2s = representations_cos_average(t2s_rep, s2s_rep)
            cos_t2t_s2s = representations_cos_average(t2t_rep, s2s_rep)
            logger.info("Decoder layer:{} Cos s2t-s2s:{} Cos t2s-s2s:{} Cos t2t-s2s:{}".format(layer_id, cos_s2t_s2s, cos_t2s_s2s, cos_t2t_s2s))

            # draw tsne
            cated_rep = torch.cat([s2t_rep, s2s_rep, t2s_rep, t2t_rep], dim=0).cpu().numpy()
            labels = [0 for i in range(len(src_sent))] + [1 for i in range(len(src_sent))] + \
                [2 for i in range(len(src_sent))] + [3 for i in range(len(src_sent))]
            tsne(cated_rep, labels, os.path.join(tsne_saved_dir, "eval_encoder_decoder_decoder_layer{}".format(layer_id)))

        logger.info("Eval encoder decoder: Done.")


def main(params):

    # initialize the experiment
    initialize_exp(params)

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
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

    evaluator = MassRepEvaluator(encoder, decoder, params, dico, "avg")
    evaluator.eval_encoder(params.src_lang, params.tgt_lang, params.src_text, params.tgt_text, params.dumped_path)
    evaluator.eval_encoder_decoder(params.src_lang, params.tgt_lang, params.src_text, params.tgt_text, params.dumped_path)

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
