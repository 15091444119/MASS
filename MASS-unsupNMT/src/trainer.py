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

from .utils import get_optimizer, to_cuda, concat_batches
from .utils import parse_lambda_config, update_lambdas

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

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params)

    def get_optimizer_fp(self, module):
        """
        Build optimizer.
        """
        optimizer = get_optimizer(getattr(self, module).parameters(), self.params.optimizer)
        if self.params.fp16:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
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
        update_lambdas(self.params, self.n_total_iter)
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
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0

        # trained combiner words
        for lang in self.params.explicit_mass_steps:
            s_speed += "- combiner {} {} words/s ".format(lang, self.stats["trained_combiner_words-{}".format(lang)] * 1.0 / diff)
            self.stats["trained_combiner_words-{}".format(lang)] = 0

        self.last_time = new_time
        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def get_iterator(self, iter_name, lang1, lang2, stream, back):
        """
        Create a new iterator for a dataset.
        """
        logger.info("Creating new training data iterator (%s) ..." % ','.join([str(x) for x in [iter_name, lang1, lang2] if x is not None]))
        if lang2 is None:
            if stream:
                iterator = self.data['mono_stream'][lang1]['train'].get_iterator(shuffle=True)
            else:
                iterator = self.data['mono'][lang1]['train'].get_iterator(
                    shuffle=True,
                    group_by_size=self.params.group_by_size,
                    n_sentences=-1,
                )
        elif back is True:
            iterator = self.data['back'][(lang1, lang2)].get_iterator(
                shuffle=True,
                group_by_size=self.params.group_by_size,
                n_sentences=-1,
            )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data['para'][(_lang1, _lang2)]['train'].get_iterator(
                shuffle=self.params.debug_train,
                group_by_size=self.params.group_by_size,
                n_sentences=-1,
            )
        logger.info("iterator (%s) done" % ','.join([str(x) for x in [iter_name, lang1, lang2] if x is not None]))

        self.iterators[(iter_name, lang1, lang2)] = iterator
        return iterator

    def get_batch(self, iter_name, lang1, lang2=None, stream=False, back=False):
        """
        Return a batch of sentences from a dataset.
        """
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None
        iterator = self.iterators.get((iter_name, lang1, lang2), None)
        if iterator is None:
            iterator = self.get_iterator(iter_name, lang1, lang2, stream, back)
        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name, lang1, lang2, stream, back)
            x = next(iterator)
        return x if lang2 is None or lang1 < lang2 else x[::-1]


    def generate_batch(self, lang1, lang2, name):
        """
        Prepare a batch (for causal or non-causal mode).
        """
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None

        if lang2 is None:
            x, lengths = self.get_batch(name, lang1, stream=True)
            positions = None
            langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
        elif lang1 == lang2:
            (x1, len1) = self.get_batch(name, lang1)
            (x2, len2) = (x1, len1)
            (x1, len1) = self.add_noise(x1, len1)
            x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=False)
        else:
            (x1, len1), (x2, len2) = self.get_batch(name, lang1, lang2)
            x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=True)

        return x, lengths, positions, langs, (None, None) if lang2 is None else (len1, len2)

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

        data['dico_id2word'] = self.data['dico'].id2word
        data['dico_word2id'] = self.data['dico'].word2id
        data['dico_counts'] = self.data['dico'].counts
        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        logger.info("Saving checkpoint to %s ..." % checkpoint_path)
        torch.save(data, checkpoint_path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = self.params.checkpoint_path
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


class Seq2SeqTrainer(Trainer):

    def __init__(self, seq2seq_model: BaseSeq2Seq, data, params):

        self.MODEL_NAMES = ['seq2seq_model']

        # model / data / params
        self.seq2seq_model = seq2seq_model
        self.data = data
        self.params = params

        # optimizers
        self.optimizers = {
            'seq2seq_model': self.get_optimizer_fp('seq2seq_model'),
        }

        super().__init__(data, params)
        self.init_stats()

    def optimize(self):
        self.optimizers["seq2seq_model"].step()
        self.optimizers["seq2seq_model"].zero_grad()

    def step(self):
        # combiner step
        for lang1, lang2 in self.params.mt_steps:
            self.mt_step(lang1, lang2, self.params.lambda_mt)
        for lang in self.params.mass_steps:
            self.mass_step(lang, self.params.lambda_mass)
        for lang in self.params.explicit_mass_steps:
            self.explicit_mass_step(lang, self.params.lambda_explicit_mass)


    def init_stats(self):
        params = self.params
        self.stats = OrderedDict(
            [('processed_s', 0), ('processed_w', 0)] +
            [('MT-%s-%s' % (l1, l2), []) for l1, l2 in params.mt_steps] +
            [('MASS-%s' % (l1), []) for l1 in params.mass_steps] +
            [('Explicit-MASS-decoding-%s' % l, []) for l in params.explicit_mass_steps] +
            [('Explicit-MASS-combiner-%s' % l, []) for l in params.explicit_mass_steps] +
            [('trained_combiner_words-{}'.format(l), 0) for l in params.explicit_mass_steps]
        )


    def mt_step(self, lang1, lang2, lambda_coeff):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        self.seq2seq_model.train()

        assert lang1 != lang2
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # generate batch
        (x1, len1), (x2, len2) = self.get_batch('mt', lang1, lang2)
        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        # cuda
        x1, len1, langs1, x2, len2, langs2, y, pred_mask = to_cuda(x1, len1, langs1, x2, len2, langs2, y, pred_mask)

        encoder_inputs = EncoderInputs(x1=x1, len1=len1, lang_id=lang1_id, langs1=langs1)
        decoder_inputs = DecoderInputs(x2=x2, len2=len2, y=y, pred_mask=pred_mask, positions=None, langs2=langs2, lang_id=lang2_id)

        _, loss = self.seq2seq_model("seq2seq_loss", encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)

        self.stats[('MT-%s-%s' % (lang1, lang2))].append(loss.item())
        loss = lambda_coeff * loss

        # optimize
        self.backward(loss, "seq2seq_model")

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += (len2 - 1).sum().item()


    def mass_step(self, lang, lambda_coeff):
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params

        self.seq2seq_model.train()

        encoder_inputs, decoder_inputs = self.get_mass_batch("mass", lang)

        _, loss = self.seq2seq_model("seq2seq_loss", encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)

        self.stats[('MASS-%s' % lang)].append(loss.item())
        loss = loss * lambda_coeff

        self.backward(loss, "seq2seq_model")

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += decoder_inputs.len2.size(0)
        self.stats['processed_w'] += (decoder_inputs.len2 - 1).sum().item()

    def explicit_mass_step(self, lang, lambda_coeff):
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params

        self.seq2seq_model.train()

        encoder_inputs, decoder_inputs = self.get_mass_batch("mass", lang)
        _, decoding_loss, combiner_loss, trained_combiner_words = self.seq2seq_model("explicit_loss", encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)

        assert combiner_loss != None

        self.stats[('Explicit-MASS-decoding-%s' % lang)].append(decoding_loss.item())
        self.stats[('Explicit-MASS-combiner-%s' % lang)].append(combiner_loss.item())

        loss = lambda_coeff * (decoding_loss + combiner_loss)
        self.backward(loss, "seq2seq_model")

        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += decoder_inputs.len2.size(0)
        self.stats['processed_w'] += (decoder_inputs.len2 - 1).sum().item()
        self.stats['trained_combiner_words-{}'.format(lang)] += trained_combiner_words


    def get_mass_batch(self, step_name, lang):
        """
        get a mass batch
        Args:
            step_name:
            lang:

        Returns:

        """
        x_, len_ = self.get_batch(step_name, lang)
        rng = np.random.RandomState()
        (x1, len1, x2, len2, y, pred_mask, positions) = mask_sent(x_, len_, rng, self.params, self.data["dico"])
        (x1, len1, x2, len2, y, pred_mask, positions) = to_cuda(x1, len1, x2, len2, y, pred_mask, positions)
        lang1_id = self.params.lang2id[lang]
        lang2_id = lang1_id
        langs1 = x1.clone().fill_(lang1_id)
        langs2 = x2.clone().fill_(lang2_id)
        encoder_inputs = EncoderInputs(x1=x1, len1=len1, lang_id=lang1_id, langs1=langs1)
        decoder_inputs = DecoderInputs(x2=x2, len2=len2, langs2=langs2, y=y, pred_mask=pred_mask, positions=positions, lang_id=lang2_id)
        return encoder_inputs, decoder_inputs


def mask_sent(x, lengths, rng, params, dico):
    def random_start(end):
        p = rng.rand()
        if p >= 0.8:
            return 1
        elif p >= 0.6:
            return end - 1
        else:
            return rng.randint(1, end)

    def mask_word(w):
        return params.mask_index

    positions, inputs, targets, outputs, len2 = [], [], [], [], []
    for i in range(lengths.size(0)):
        words = x[:lengths[i], i].tolist()
        l = len(words)
        # Prevent some short sentences will be whole masked
        mask_len = max(1, round(l * params.word_mass) - 1)
        start = random_start(l - mask_len + 1)

        if params.encoder_type == "combiner":  # we have to do whole word mask because the splitter
            # whole word mask
            while (start - 1 >= 0 and "@@" in dico.id2word[x[start - 1, i].item()]):
                start -= 1
            end = start + mask_len - 1

            while (end + 1 < lengths[i].item() and "@@" in dico.id2word[x[end, i].item()]):
                end += 1
        else:
            end = start + mask_len - 1

        len2.append(end - start + 1)

        pos_i, target_i, output_i, input_i = [], [], [], []
        prev_w = None
        for j, w in enumerate(words):
            if j >= start and j <= end:
                output_i.append(w)
                target_i.append(prev_w)
                pos_i.append(j - 1)
                input_i.append(mask_word(w))
            else:
                input_i.append(w)
            prev_w = w

        inputs.append(input_i)
        targets.append(target_i)
        outputs.append(output_i)
        positions.append(pos_i)

    l1 = lengths.clone()
    l2 = torch.LongTensor(len2)
    x1 = torch.LongTensor(max(l1), l1.size(0)).fill_(params.pad_index)
    x2 = torch.LongTensor(max(len2), l1.size(0)).fill_(params.pad_index)
    y = torch.LongTensor(max(len2), l1.size(0)).fill_(params.pad_index)
    pos = torch.LongTensor(max(len2), l1.size(0)).fill_(params.pad_index)

    for i in range(l1.size(0)):
        x1[:l1[i], i].copy_(torch.LongTensor(inputs[i]))
        x2[:l2[i], i].copy_(torch.LongTensor(targets[i]))
        y[:l2[i], i].copy_(torch.LongTensor(outputs[i]))
        pos[:l2[i], i].copy_(torch.LongTensor(positions[i]))
    pred_mask = y != params.pad_index
    y = y.masked_select(pred_mask)

    return x1, l1, x2, l2, y, pred_mask, pos
