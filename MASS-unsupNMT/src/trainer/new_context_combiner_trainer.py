# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#

import os
import numpy
import math
import pdb
import time
import random
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from apex.fp16_utils import FP16_Optimizer
from src.model.seq2seq import BaseSeq2Seq, DecoderInputs
from src.model.encoder import EncoderInputs

from src.utils import get_optimizer, to_cuda, concat_batches
from src.utils import parse_lambda_config, update_lambdas

logger = getLogger()


class Trainer(object):

    def __init__(self, data, params):
        """
        Initialize trainer.
        """
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # data iterators
        self.iterators = {}

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0

        self.last_time = time.time()

        # reload potential checkpoints
        if params.checkpoint != "":
            self.reload_checkpoint()


    def get_optimizer_fp(self, trained_parameters):
        """
        Build optimizer.
        """
        optimizer = get_optimizer(trained_parameters, self.params.optimizer)
        optimizer.zero_grad()
        return optimizer

    def backward(self, loss, module):
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        loss.backward()
        if self.params.clip_grad_norm > 0:
            clip_grad_norm_(getattr(self, module).parameters(), self.params.clip_grad_norm)

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """

        if self.n_iter % 5 != 0:
            return

        s_iter = "%7i - " % self.n_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # transformer learning rate
        lr = self.optimizers[self.MODEL_NAMES[0]].param_groups[0]['lr']
        s_lr = " - Transformer LR = {:.4e}".format(lr)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s ".format(
            self.stats['processed_s'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff,
        )

        s_acc = " train_acc:{} ".format(self.stats['right_w'] / self.stats['processed_w'])
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0
        self.stats['right_w'] = 0

        self.last_time = new_time
        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr + s_acc)


    def save_model(self, name):
        """
        Save the model.
        """
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info('Saving models to %s ...' % path)
        data = {}
        for name in self.MODEL_NAMES:
            if self.params.multi_gpu:
                data[name] = getattr(self, name).module.state_dict()
            else:
                data[name] = getattr(self, name).state_dict()

        data['dico_id2word'] = self.data['dico'].id2word
        data['dico_word2id'] = self.data['dico'].word2id
        data['dico_counts'] = self.data['dico'].counts
        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        if not self.params.is_master:
            return

        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
        }

        for name in self.MODEL_NAMES:
            data[name] = getattr(self, name).state_dict()
            data[name + '_optimizer'] = self.optimizers[name].state_dict()

        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        logger.info("Saving checkpoint to %s ..." % checkpoint_path)
        torch.save(data, checkpoint_path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = self.params.checkpoint
        assert os.path.isfile(checkpoint_path)
        logger.warning('Reloading checkpoint from %s ...' % checkpoint_path)
        data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(self.params.local_rank))

        # reload model parameters and optimizers
        for name in self.MODEL_NAMES:
            getattr(self, name).load_state_dict(data[name])
            self.optimizers[name].load_state_dict(data[name + '_optimizer'])

        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']
        self.best_metrics = data['best_metrics']
        self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_model('periodic-%i' % self.epoch)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_model('best-%s' % metric)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (self.params.is_master or not self.stopping_criterion[0].endswith('_mt_bleu')):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                    os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                exit()
        self.save_checkpoint()
        self.epoch += 1


class NewContextCombinerTrainer(Trainer):

    def __init__(self, encoder, combiner, pred_layer, data, loss_fn, params, lang_id):

        super().__init__(data, params)
        self.MODEL_NAMES = ['combiner']

        # model / data / params
        self.combiner = combiner
        self.encoder = encoder
        self.pred_layer = pred_layer
        self.data = data
        self.loss_fn = loss_fn
        self.params = params
        self.lang_id = lang_id

        self.optimizers = {
            'combiner': self.get_optimizer_fp(self.combiner.parameters()),
        }

        self.init_stats()

        self.iterators["combine"] = iter(self.data["train"])

    def get_batch(self):
        try:
            batch = next(self.iterators["combine"])
        except StopIteration:
            self.iterators["combine"] = iter(self.data["train"])
            batch = next(self.iterators["combine"])
        return batch

    def optimize(self):
        self.optimizers["combiner"].step()
        self.optimizers["combiner"].zero_grad()
        self.combiner.zero_grad()

    def step(self):

        batch = self.get_batch()

        self.encoder.eval()
        self.combiner.train()
        self.pred_layer.eval()
        loss, results, trained_sentences, trained_words = combiner_step(
            encoder=self.encoder,
            combiner=self.combiner,
            lang_id=self.lang_id,
            loss_fn=self.loss_fn,
            batch=batch,
            pred_layer=self.pred_layer
        )

        loss = loss.mean()

        self.stats['combiner-loss'].append(loss.item())
        self.backward(loss, "combiner")
        self.stats['processed_s'] += trained_sentences
        self.stats['right_w'] += results.long().sum().item()
        self.stats['processed_w'] += trained_words
        self.n_sentences += trained_sentences

    def init_stats(self):
        self.stats = OrderedDict(
            [('processed_s', 0), ('processed_w', 0), ('right_w', 0)] +
            [('combiner-loss', [])]
        )


def combiner_step(encoder, combiner, pred_layer, lang_id, batch, loss_fn, debug_dict=None):
    original_batch = batch["original_batch"].cuda()
    original_length = batch["original_length"].cuda()
    splitted_batch = batch["splitted_batch"].cuda()
    splitted_length = batch["splitted_length"].cuda()
    trained_word_mask = batch["trained_word_mask"].cuda()

    combine_labels = batch["combine_labels"].cuda()

    original_batch_langs = original_batch.clone().fill_(lang_id).cuda()
    splitted_batch_langs = splitted_batch.clone().fill_(lang_id).cuda()

    with torch.no_grad():
        splitted_encoded = encoder(
            "fwd",
            x=splitted_batch,
            lengths=splitted_length,
            langs=splitted_batch_langs,
            causal=False
        ).transpose(0, 1)  # [bs, len, dim]

    if debug_dict is not None:
        slen, bs = original_batch.size()
        for i in range(bs):
            for j in range(slen):
                token = debug_dict.id2word[original_batch[j][i].item()]
                if trained_word_mask[i][j] == True:
                    token = '[[' + token + "]]"
                print(token, end=" ")
            print()

        slen, bs = splitted_batch.size()
        for i in range(bs):
            for j in range(slen):
                token = debug_dict.id2word[splitted_batch[j][i].item()]
                if combine_labels[i][j].item() in [2, 3]:
                    token = '[[' + token + "]]"
                print(token, end=" ")

    combined_rep = combiner(splitted_encoded, splitted_length, combine_labels)

    y = original_batch.transpose(0, 1).masked_select(trained_word_mask)
    scores, loss = pred_layer(combined_rep, y)

    results = scores.argmax(dim=-1) == y

    trained_sentences = original_batch.size(1)

    trained_words = trained_word_mask.long().sum().item()

    assert loss.size(0) == results.size(0)
    assert loss.size(0) == trained_words
    assert loss.size(0) == trained_sentences

    return loss, results, trained_sentences, trained_words
