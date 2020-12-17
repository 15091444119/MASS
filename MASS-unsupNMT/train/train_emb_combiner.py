"""

Train emb combiner
1. Load mass dico and embeddings (if need to evaluate mass performance, also load mass)

2. Load combiner train, dev, test loader

3. Train combiner, and evaluate it

"""

import argparse
import os
import tensorboardX
from src.utils import bool_flag, initialize_exp
from src.data.emb_combiner_data.emb_combiner_dataloader import build_emb_combiner_dataloader
from src.model import build_loss_function
from src.evaluation.emb_combiner_evaluator import EmbCombinerEvaluator
from src.data.splitter import WholeWordSplitter
from src.evaluation.utils import load_mass_model
import json
from src.model.combiner.emb_combiner import build_emb_combiner_model
from src.utils import bool_flag
from src.trainer.emb_combiner_trainer import EmbCombinerTrainer


def parse_args():
    parser = argparse.ArgumentParser("parameters for emb combiner")

    parser.add_argument("--dump_path")
    parser.add_argument("--exp_id")
    parser.add_argument("--exp_name")

    # only eval
    parser.add_argument("--eval_only", type=bool_flag)

    # train params
    parser.add_argument("--eval_loss_samples", type=int)
    parser.add_argument("--max_epoch", type=int)
    parser.add_argument("--epoch_size", type=int)
    parser.add_argument("--stopping_criterion", type=str)
    parser.add_argument("--validation_metrics", type=str)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--clip_grad_norm", type=float, default=5)

    # embedding
    parser.add_argument("--mass_model", type=str)

    # data
    parser.add_argument("--train", type=str)
    parser.add_argument("--dev", type=str)
    parser.add_argument("--test", type=str)
    parser.add_argument("--batch_size", type=int)

    # splitter parameters
    parser.add_argument("--splitter", type=str, choices=["BPE", "CHAR", "ROB"], help="use random bpe or character to split word")
    parser.add_argument("--codes_path", type=str)

    # combiner parameters
    parser.add_argument("--combiner_type", type=int)
    parser.add_argument("--context_extractor_type", type=str)

    if parser.parse_known_args()[0].combiner_type in ["GRU", "TRANSFORMER"]:
        parser.add_argument("--n_combiner_layer", type=int)

    if parser.parse_known_args()[0].combiner_type in ["TRANSFORMER"]:
        parser.add_argument("--n_head", type=int)

    args = parser.parse_args()

    return args


def build_data(args, dico):
    train_dataloader = build_emb_combiner_dataloader(
        whole_word_path=args.train,
        dico=dico,
        batch_size=args.batch_size,
        shuffle=True
    )

    dev_dataloader = build_emb_combiner_dataloader(
        whole_word_path=args.dev,
        dico=dico,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_dataloader = build_emb_combiner_dataloader(
        whole_word_path=args.test,
        dico=dico,
        batch_size=args.batch_size,
        shuffle=True
    )

    data = {
        "train": train_dataloader,
        "dev": dev_dataloader,
        "test": test_dataloader,
        "dico": dico
    }

    return data



def main():

    args = parse_args()

    logger = initialize_exp(args)

    dico, mass_params, encoder, decoder = load_mass_model(args)

    emb_dim = mass_params.emb_dim

    embeddings = encoder.embeddings

    splitter = WholeWordSplitter.build_splitter(
        splitter=args.splitter,
        codes_path=args.codes_path,
        word_vocab=dico.word2id.keys()
    )

    data = build_data(args=args, dico=dico)

    model = build_emb_combiner_model(emb_dim=emb_dim, params=args)

    loss = build_loss_function(args.loss)

    trainer = EmbCombinerTrainer(data=data, combiner=model, params=args, embeddings=embeddings, loss_fn=loss)


    evaluator = EmbCombinerEvaluator(data=data, params=args, combiner=model, embeddings=embeddings, loss_fn=loss)


    # evaluation
    if args.eval_only:
        scores = evaluator.run_all_evals(trainer.epoch)
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # summary writer
    writer = tensorboardX.SummaryWriter(os.path.join(args.dump_path, 'tensorboard'))

    # training
    for _ in range(args.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_sentences = 0
        last_eval_loss_sentences = 0

        while trainer.n_sentences < trainer.epoch_size:

            trainer.step()
            trainer.iter()

            # evaluate loss
            if args.eval_loss_sentences != -1 and trainer.n_sentences - last_eval_loss_sentences >= args.eval_loss_sentences:
                scores = {}
                evaluator.eval_loss(scores)
                for k, v in scores.items():
                    logger.info("%s -> %.6f" % (k, v))
                logger.info("__log__:%s" % json.dumps(scores))
                last_eval_loss_sentences = trainer.n_sentences

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate
        scores = evaluator.run_all_evals(trainer.epoch)

        # print / JSON log
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))

        # save log in tensorboard
        for k, v in scores.items():
            writer.add_scalar(k, v, trainer.epoch)

        # end of epoch
        trainer.save_best_model(scores)
        trainer.end_epoch(scores)
