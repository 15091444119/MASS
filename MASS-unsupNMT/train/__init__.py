import tensorboardX
import json
import os
import logging

logger = logging.getLogger()


def single_gpu_run(trainer, evaluator, eval_only, dump_path, eval_loss_sentences, max_epoch):
    """

    Args:
        trainer:
        evaluator:
        eval_only:
        dump_path:
        eval_loss_sentences:
        max_epoch:

    Returns:

    """

