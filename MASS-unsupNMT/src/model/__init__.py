# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#

from logging import getLogger
import os
import torch

from .transformer import TransformerModel
from src.model.seq2seq.common_seq2seq import CommonSeq2Seq
from src.model.seq2seq.combiner_seq2seq import CombinerEncoder, CombinerSeq2Seq
from src.data.splitter import WholeWordSplitter
from src.model.context_combiner.context_combiner import build_combiner

from collections import OrderedDict
from src.data.dictionary import Dictionary
from src.utils import AttrDict

logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.sample_alpha < 1

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # reload a pretrained model or reload from checkpoint
    assert not (params.reload_model != '' and params.checkpoint != '')  # checkpoint and reload model can't be used at the same time
    if params.reload_model != '':
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            s = params.reload_model.split(',')
            assert len(s) == 2
            assert all([x == '' or os.path.isfile(x) for x in s])


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


def build_model(params, dico):
    """
    Build model.
    """
    # build
    encoder = TransformerModel(params, dico, is_encoder=True, with_output=True)  # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
    decoder = TransformerModel(params, dico, is_encoder=False, with_output=True)

    # reload a mass pretrained model
    if params.reload_model != '':
        enc_path, dec_path = params.reload_model.split(',')
        assert not (enc_path == '' and dec_path == '')

        # reload encoder
        if enc_path != '':
            logger.info("Reloading encoder from %s ..." % enc_path)
            enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
            enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
            if all([k.startswith('module.') for k in enc_reload.keys()]):
                enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}
            encoder.load_state_dict(enc_reload)

        # reload decoder
        if dec_path != '':
            logger.info("Reloading decoder from %s ..." % dec_path)
            dec_reload = torch.load(dec_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
            dec_reload = dec_reload['model' if 'model' in dec_reload else 'decoder']
            if all([k.startswith('module.') for k in dec_reload.keys()]):
                dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}
            decoder.load_state_dict(dec_reload, strict=False)

    if params.encoder_type == "common":
        encoder = CommonEncoder(encoder)
    elif params.encoder_type == "combiner":
        encoder = build_combiner_encoder(encoder=encoder, params=params, dico=dico)
    else:
        raise ValueError


    logger.info("Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
    logger.info("Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))

    if params.encoder_type == "common":
        seq2seq_model = CommonSeq2Seq(encoder=encoder, decoder=decoder)
    elif params.encoder_type == "combiner":
        seq2seq_model = CombinerSeq2Seq(encoder=encoder, decoder=decoder)
    logger.info("Model:{}".format(seq2seq_model))

    return seq2seq_model.cuda()


def build_loss_function(loss, reduction="average"):
    """

    Args:
        loss:
        reduction:
            if "average": a batch averaged loss will be returned
            if "batch": return a tensor of size [batch_size]

    Returns:
        loss_fn

    """
    def loss_fn(x, y):
        if loss == "MSE":
            batch_loss = torch.nn.MSELoss(reduction="none")(x, y).mean(dim=1)
        elif loss == "COS":
            batch_loss = -torch.nn.CosineSimilarity(dim=1)(x, y)
        else:
            raise NotImplementedError

        if reduction == "average":
            return batch_loss.mean()
        elif reduction == "batch":
            return batch_loss
        else:
            raise NotImplementedError

    return loss_fn


def build_combiner_encoder(encoder, params, dico):
    loss_fn = build_loss_function(params)
    splitter = WholeWordSplitter.build_splitter(splitter=params.splitter, codes_path=params.codes_path, word_vocab=dico.word2id.keys())
    combiner = build_combiner(params)

    combiner_encoder = CombinerEncoder(
        encoder=encoder,
        combiner=combiner,
        params=params,
        dico=dico,
        splitter=splitter,
        loss_fn=loss_fn
    )

    return combiner_encoder


def load_combiner_model(model_path):
    print("Load model from {}".format(model_path))
    reloaded = torch.load(model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])

    assert model_params.encoder_type == "combiner"
    combiner_seq2seq = build_model(model_params, dico)

    combiner_seq2seq.load_state_dict(package_module(reloaded["seq2seq_model"]))

    return dico, model_params, combiner_seq2seq


def package_module(modules):
    """
    Return model state dict, take multi-gpu case into account
    """
    state_dict = OrderedDict()
    for k, v in modules.items():
        if k.startswith('module.'):
            state_dict[k[7:]] = v
        else:
            state_dict[k] = v
    return state_dict
