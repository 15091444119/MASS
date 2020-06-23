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

def get_all_layer_representation(encoder, decoder, params, dico, src_lang, tgt_lang, path, method='avg'):
    # read sentences from stdin
    src_sent = []
    with open(path, 'r') as f:
        for line in f:
            assert len(line.strip().split()) > 0
            src_sent.append(line)
            if len(src_sent) == 100:
                break
        logger.info("Read %{} sentences from {}. ".format(len(src_sent), path))
    
    all_layer_sen_representation = [[] for i in range(encoder.n_layers + 1)] 

    for i in range(0, len(src_sent), params.batch_size):

        # prepare batch
        word_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                    for s in src_sent[i:i + params.batch_size]]
        lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(params.pad_index)
        batch[0] = params.eos_index
        for j, s in enumerate(word_ids):
            if lengths[j] > 2:  # if sentence not empty
                batch[1:lengths[j] - 1, j].copy_(s)
            batch[lengths[j] - 1, j] = params.eos_index
        langs = batch.clone().fill_(params.lang2id[src_lang])

        # encode source batch and translate it
        encoded, _ = encoder.get_all_layer_outputs(x=batch.cuda(), lengths=lengths.cuda(), langs=langs.cuda(), causal=False)
        
        for layer_id, layer_encoded in enumerate(encoded):
            layer_sen_representation = get_sen_representation(layer_encoded, lengths.cuda(), method)
            all_layer_sen_representation[layer_id].append(layer_sen_representation)
        
    for layer_id in range(len(all_layer_sen_representation)):
        all_layer_sen_representation[layer_id] = torch.cat(all_layer_sen_representation[layer_id], dim=0)

    return src_sent, all_layer_sen_representation

def representations_cos_average(src_representation, tgt_representation):
    """ Average cos similarity of source sentences and target sentences """
    assert src_representation.size(0) == tgt_representation.size(0)
    logger.info("{}".format(src_representation.size(0)))
    #tgt_representation_reverse =torch.clone(tgt_representation)
    #for i in range(tgt_representation.size(0)):
    #    tgt_representation_reverse[i] = tgt_representation[tgt_representation.size(0) - 1 - i]
    cos_simi = torch.nn.functional.cosine_similarity(src_representation, tgt_representation, dim=-1)
    return cos_simi.mean().item()

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


    src_sents, src_all_layer_representation = get_all_layer_representation(encoder, decoder, params, dico, params.src_lang, params.tgt_lang, params.src_text) 
    tgt_sents, tgt_all_layer_representation = get_all_layer_representation(encoder, decoder, params, dico, params.tgt_lang, params.src_lang, params.tgt_text) 

    for layer_id, (src_layer_representation, tgt_layer_representation) in enumerate(zip(src_all_layer_representation, tgt_all_layer_representation)):
        avg_cos = representations_cos_average(src_layer_representation, tgt_layer_representation)
        logger.info("Layer{} average cos similarity:{}".format(layer_id, avg_cos))
        """
        src_tgt_representation = torch.cat([src_layer_representation, tgt_layer_representation], dim=0).cpu().numpy()
        labels = [0 for i in range(src_layer_representation.size(0))] + [1 for i in range(tgt_layer_representation.size(0))]
        tsne(src_tgt_representation, labels, "./tmp1000-{}".format(layer_id))
        """
        bilingual_tsne(src_layer_representation.cpu().numpy(), tgt_layer_representation.cpu().numpy(), src_sents, tgt_sents, 5, "./5-{}".format(layer_id))

    logger.info("Done")
    


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
