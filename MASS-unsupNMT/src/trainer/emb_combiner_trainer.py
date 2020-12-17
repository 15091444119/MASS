import os
import time
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from logging import getLogger
from src.utils import get_optimizer, to_cuda, concat_batches
from collections import OrderedDict

logger = getLogger()


class EmbCombinerBaseTrainer(object):

    def __init__(self, data, params):
        """
        Initialize trainer.
        """
        self.data = data
        self.params = params

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

    def get_optimizer_fp(self, trained_parameters):
        """
        Build optimizer.
        """
        optimizer = get_optimizer(trained_parameters, self.params.optimizer)
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

        if self.n_iter % 10 == 0:
            self.print_stats()

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

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
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
        if self.stopping_criterion is not None:
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
                exit()
        self.save_checkpoint()
        self.epoch += 1


class EmbCombinerTrainer(EmbCombinerBaseTrainer):
    def __init__(self, combiner, embeddings, data, params, loss_fn):

        super().__init__(data, params)

        self.MODEL_NAMES = ['combiner']

        # model / data / params
        self.combiner = combiner
        self.emebddings = embeddings

        self.loss_fn = loss_fn

        self.optimizers = {
            'combiner': self.get_optimizer_fp(combiner.parameters()),
        }

        self.init_stats()

        self.iterator = iter(self.data["train"])

    def optimize(self):
        self.optimizers["combiner"].step()
        self.optimizers["combiner"].zero_grad()
        self.combiner.zero_grad()

    def get_batch(self):

        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.data["train"])
            batch = next(self.iterator)

        return batch

    def step(self):
        self.combiner.train()
        self.emebddings.eval()

        batch = self.get_batch()
        batch_splitted_word_ids, splitted_words_lengths, batch_whole_word_id = batch

        combiner_rep = self.combiner(batch_splitted_word_ids, splitted_words_lengths)

        original_rep = self.emebddings(batch_whole_word_id).detach()  # don't update embeddings

        loss = self.loss_fn(combiner_rep, original_rep)

        self.stats["combiner_training_loss"].append(loss.item())
        self.stats["processed_w"] += batch_whole_word_id.size(0)
        self.n_sentences += batch_whole_word_id.size(0)

        self.backward(loss, self.combiner)
        self.optimize()

    def init_stats(self):
        self.stats = OrderedDict(
            [('processed_w', 0)] +
            [("combiner_training_loss", [])]
        )

    def print_stats(self):
        """
        Print statistics about the training.
        """

        # stats to be averaged
        s_iter = "%7i - " % self.n_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rate
        lr = self.optimizers[self.MODEL_NAMES[0]].param_groups[0]['lr']
        s_lr = " - LR = {:.4e}".format(lr)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:8.2f} words/s ".format(
            self.stats['processed_w'] * 1.0 / diff,
            )
        self.stats['processed_w'] = 0

        self.last_time = new_time
        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)