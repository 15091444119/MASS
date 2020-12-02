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
from src.model.seq2seq.common_seq2seq import CommonSeq2Seq, CommonEncoder


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

    logger.debug("Encoder: {}".format(encoder))
    logger.debug("Decoder: {}".format(decoder))
    logger.info("Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
    logger.info("Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))

    encoder = CommonEncoder(encoder)
    seq2seq_model = CommonSeq2Seq(encoder=encoder, decoder=decoder)

    return seq2seq_model.cuda()




