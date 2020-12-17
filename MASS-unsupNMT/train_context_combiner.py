# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#

import json
import argparse
import tensorboardX
import torch
import os

from src.slurm import init_signal_handler, init_distributed_mode
from src.data.loader import check_data_params, load_data, check_eval_params
from src.utils import bool_flag, initialize_exp
from src.model import check_model_params, build_model
from src.trainer import  Seq2SeqTrainer
from src.evaluation.evaluator import Seq2SeqEvaluator


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")

    # only use an encoder (use a specific decoder for machine translation)
    parser.add_argument("--encoder_only", type=bool_flag, default=False,
                        help="Only use an encoder")
    parser.add_argument("--english_only", type=bool_flag, default=False,
                        help="Only use english domain (equal to only use one language)")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument("--n_dec_layers", type=int, default=6,
                        help="Number of Decoder Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    parser.add_argument("--attention_setting", type=str, default="v1", choices=["v1", "v2"],
                        help="Setting for attention module, benefits for distinguish language")

    # adaptive softmax
    parser.add_argument("--asm", type=bool_flag, default=False,
                        help="Use adaptive softmax")
    if parser.parse_known_args()[0].asm:
        parser.add_argument("--asm_cutoffs", type=str, default="8000,20000",
                            help="Adaptive softmax cutoffs")
        parser.add_argument("--asm_div_value", type=float, default=4,
                            help="Adaptive softmax cluster sizes ratio")

    # causal language modeling task parameters
    parser.add_argument("--context_size", type=int, default=0,
                        help="Context size (0 means that the first elements in sequences won't have any context)")

    # masked language modeling task parameters
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="Fraction of words for which we need to make a prediction")
    parser.add_argument("--sample_alpha", type=float, default=0,
                        help="Exponent for transforming word counts to probabilities (~word2vec sampling)")

    parser.add_argument("--word_mass", type=float, default=0,
                        help="Randomly mask input words (0 to disable)")

    # data
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path")
    parser.add_argument("--lgs", type=str, default="",
                        help="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")
    parser.add_argument("--lg_sampling_factor", type=float, default=-1,
                        help="Language sampling factor")
    parser.add_argument("--n_para_train", type=int, default=-1, help="Number of data used for parallel training")

    # batch parameters
    parser.add_argument("--bptt", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--min_len", type=int, default=0,
                        help="Minimum length of sentences (after BPE)")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--group_by_size", type=bool_flag, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")

    # training parameters
    parser.add_argument("--split_data", type=bool_flag, default=False,
                        help="Split data across workers of a same node")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics, if start with _, the smaller the better")

    # training coefficients
    parser.add_argument("--lambda_mlm", type=str, default="1",
                        help="Prediction coefficient (MLM)")
    parser.add_argument("--lambda_clm", type=str, default="1",
                        help="Causal coefficient (LM)")
    parser.add_argument("--lambda_bmt", type=str, default="1",
                        help="Back Parallel coefficient")
    parser.add_argument("--lambda_pc", type=str, default="1",
                        help="PC coefficient")
    parser.add_argument("--lambda_ae", type=str, default="1",
                        help="AE coefficient")
    parser.add_argument("--lambda_mt", type=str, default="1",
                        help="MT coefficient")
    parser.add_argument("--lambda_bt", type=str, default="1",
                        help="BT coefficient")
    parser.add_argument("--lambda_mass", type=str, default="1",
                        help="MASS coefficient")
    parser.add_argument("--lambda_span", type=str, default="10000",
                        help="Span coefficient")
    parser.add_argument("--lambda_explicit_mass", type=str, default="1",
                        help="MASS and combine coefficient")

    # training steps
    parser.add_argument("--clm_steps", type=str, default="",
                        help="Causal prediction steps (CLM)")
    parser.add_argument("--mlm_steps", type=str, default="",
                        help="Masked prediction steps (MLM / TLM)")
    parser.add_argument("--bmt_steps", type=str, default="",
                        help="Back Machine Translation step")
    parser.add_argument("--mass_steps", type=str, default="",
                        help="MASS prediction steps")
    parser.add_argument("--mt_steps", type=str, default="",
                        help="Machine translation steps")
    parser.add_argument("--ae_steps", type=str, default="",
                        help="Denoising auto-encoder steps")
    parser.add_argument("--bt_steps", type=str, default="",
                        help="Back-translation steps")
    parser.add_argument("--pc_steps", type=str, default="",
                        help="Parallel classification steps")
    parser.add_argument("--explicit_mass_steps", type=str, default="")

    # reload a pretrained model
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")

    # reload a checkpoint
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # evaluation
    parser.add_argument("--eval_bleu", type=bool_flag, default=False,
                        help="Evaluate BLEU score during MT training")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")

    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    # encoder mode
    parser.add_argument("--encoder_type", type=str, choices=["common", "combiner"])

    # combiner
    parser.add_argument("--combiner", type=str)
    parser.add_argument("--splitter", type=str, choices=["BPE", "CHAR", "ROB"], help="use random bpe or character to split word")
    parser.add_argument("--codes_path", type=str)
    parser.add_argument("--n_combiner_layers", type=int, default=4)
    parser.add_argument("--combiner_loss", type=str, default="MSE", choices=["MSE", "COS", "BNC"])
    parser.add_argument("--re_encode_rate", type=float, default=0.0)
    parser.add_argument("--train_combiner_only", type=bool_flag, default=False)


    # evaluation params
    parser.add_argument("--eval_loss_sentences", type=int, default=-1)
    parser.add_argument("--eval_mt_steps", type=str, default="")
    parser.add_argument("--eval_mass_steps", type=str, default="")
    parser.add_argument("--eval_explicit_mass_steps", type=str, default="")

    # evaluate alignment params
    parser.add_argument("--eval_alignment", type=bool_flag, default=False)
    if parser.parse_known_args()[0].eval_alignment:
        parser.add_argument("--alignment_src_bped_path", required=True)
        parser.add_argument("--alignment_tgt_bped_path", required=True)
        parser.add_argument("--alignment_path", required=True)
        parser.add_argument("--alignment_src_lang", required=True)
        parser.add_argument("--alignment_tgt_lang", required=True)
        parser.add_argument("--alignment_metric", required=True)

    # multi
    parser.add_argument("--optimize_batches", type=int, default=5)

    return parser


def main(params):
    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    # load data
    data = load_data(params)

    # build model
    seq2seq_model = build_model(params, data['dico'])

    # distributed
    if params.multi_gpu:
        logger.info("Using nn.parallel.DistributedDataParallel ...")
        seq2seq_model = torch.nn.parallel.distributed.DistributedDataParallel(seq2seq_model, device_ids=[params.local_rank], output_device=params.local_rank, find_unused_parameters=True)

    trainer = Seq2SeqTrainer(seq2seq_model, data, params)
    evaluator = Seq2SeqEvaluator(trainer=trainer, data=data, params=params)

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals(trainer.epoch)
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # summary writer
    writer = tensorboardX.SummaryWriter(os.path.join(params.dump_path, 'tensorboard'))

    optimize_count = 0
    # training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_sentences = 0
        last_eval_loss_sentences = 0

        while trainer.n_sentences < trainer.epoch_size:

            optimize_count += 1

            if optimize_count % params.optimize_batches == 0:
                trainer.step()
                trainer.optimize()
                trainer.iter()
                optimize_count = 0
            else:
                if params.multi_gpu:
                    with seq2seq_model.no_sync():
                        trainer.step()
                else:
                    trainer.step()


            # evaluate loss
            if params.eval_loss_sentences != -1 and trainer.n_sentences - last_eval_loss_sentences >= params.eval_loss_sentences:
                scores = {}
                evaluator.eval_loss(scores)
                for k, v in scores.items():
                    logger.info("%s -> %.6f" % (k, v))
                if params.is_master:
                    logger.info("__log__:%s" % json.dumps(scores))
                last_eval_loss_sentences = trainer.n_sentences


        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate perplexity
        scores = evaluator.run_all_evals(trainer.epoch)

        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        # save log in tensorboard
        for k, v in scores.items():
            writer.add_scalar(k, v, trainer.epoch)

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    check_data_params(params)
    check_eval_params(params)

    check_model_params(params)

    # run experiment
    main(params)