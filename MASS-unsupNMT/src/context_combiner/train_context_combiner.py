"""
Only train context combiner and only use encoder, don't consider decoder or mt or bli
"""

import json
import argparse
import tensorboardX
import torch
import os

from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import bool_flag, initialize_exp
from src.evaluation.utils import load_mass_model
from src.model.context_combiner.context_combiner import build_combiner
from src.context_combiner.context_combiner_trainer import NewContextCombinerTrainer
from src.context_combiner.evaluator import NewContextCombinerEvaluator
import src.context_combiner.data.dataloader as dataloader
from src.model import build_loss_function
from src.data.splitter import WholeWordSplitter


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


    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
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
    parser.add_argument("--eval_loss_sentences", type=int, default=-1)

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    # model
    parser.add_argument("--combiner", type=str)

    if parser.parse_known_args()[0].combiner in ["last_token", "word_input"]:
        parser.add_argument("--emb_dim", type=int, default=512,
                            help="Embedding layer size")
        parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False)
        parser.add_argument("--n_head", type=int)
        parser.add_argument("--combine_label_embedding", type=bool_flag, default=True)

        if parser.parse_known_args()[0].combiner == "word_input":
            parser.add_argument("--n_another_context_encoder_layer", type=int)
            parser.add_argument("--n_word_combiner_layer", type=int)
        elif parser.parse_known_args()[0].combiner == "last_token":
            parser.add_argument("--n_layer", type=int)

    # loss function
    parser.add_argument("--combiner_loss", type=str, default="MSE", choices=["MSE", "COS", "BNC"])

    # splitter
    parser.add_argument("--splitter", type=str, choices=["BPE", "CHAR", "ROB"], help="use random bpe or character to split word")
    parser.add_argument("--codes_path", type=str)

    # combiner data
    parser.add_argument("--combiner_train_data", type=str)
    parser.add_argument("--word_sample_for_train", type=bool_flag, default=True)
    parser.add_argument("--combiner_dev_data", type=str)
    parser.add_argument("--combiner_test_data", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--reload_combiner", type=str, default="")

    # multi
    parser.add_argument("--optimize_batches", type=int, default=1)

    return parser



def main(params):
    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    dico, mass_params, encoder, _ = load_mass_model(params.reload_model)
    encoder = encoder.cuda()

    splitter = WholeWordSplitter.build_splitter(splitter=params.splitter, codes_path=params.codes_path, word_vocab=dico.word2id.keys())

    data = dataloader.load_data(params=params, dico=dico, splitter=splitter)

    loss_fn = build_loss_function(params.combiner_loss, reduction="batch")

    combiner = build_combiner(params).cuda()

    if params.reload_combiner != "":
        reloaded = torch.load(params.reload_combiner)
        combiner.load_state_dict(reloaded["combiner"])

    # distributed
    if params.multi_gpu:
        logger.info("Using nn.parallel.DistributedDataParallel ...")
        combiner = torch.nn.parallel.distributed.DistributedDataParallel(combiner, device_ids=[params.local_rank], output_device=params.local_rank, find_unused_parameters=True)

    lang_id = mass_params.lang2id[params.lang]

    evaluator = NewContextCombinerEvaluator(encoder=encoder, combiner=combiner, data=data, params=params, loss_fn=loss_fn, lang_id=lang_id)

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals(0)
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    trainer = NewContextCombinerTrainer(encoder=encoder, combiner=combiner, data=data, params=params, loss_fn=loss_fn, lang_id=lang_id)

    # summary writer
    writer = tensorboardX.SummaryWriter(os.path.join(params.dump_path, 'tensorboard'))

    optimize_count = 0

    # training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_sentences = 0
        last_eval_loss_sentences = 0

        while trainer.n_sentences < trainer.epoch_size:
            print(trainer.n_sentences, trainer.epoch_size)

            optimize_count += 1

            if optimize_count % params.optimize_batches == 0:
                trainer.step()
                trainer.optimize()
                trainer.iter()
                optimize_count = 0
            else:
                if params.multi_gpu:
                    with combiner.no_sync():
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

    # run experiment
    main(params)
