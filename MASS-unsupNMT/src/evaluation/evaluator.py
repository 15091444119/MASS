# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#

from logging import getLogger
import pdb
import os
import sys
import subprocess
from collections import OrderedDict
import numpy as np
import torch

from ..utils import to_cuda, restore_segmentation, concat_batches
from .utils import SenteceEmbedder, WordEmbedderWithCombiner
from src.combiner.combiner import MultiLingualNoneParaCombiner
from .bli import BLI
from .eval_context_bli import eval_whole_separated_bli, read_retokenize_words, generate_context_word_representation, encode_whole_word_separated_word, generate_and_eval


BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH)


logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.dico = data['dico']
        self.params = params

        # create directory to store hypotheses, and reference files for BLEU evaluation
        if self.params.is_master:
            params.hyp_path = os.path.join(params.dump_path, 'hypotheses')
            subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()
            self.create_reference_files()

    def get_iterator(self, data_set, lang1, lang2=None, stream=False):
        """
        Create a new iterator for a dataset.
        """
        assert data_set in ['valid', 'test', 'train']
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None

        # hacks to reduce evaluation time when using many languages
        if len(self.params.langs) > 30:
            eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh", "ab", "ay", "bug", "ha", "ko", "ln", "min", "nds", "pap", "pt", "tg", "to", "udm", "uk", "zh_classical"])
            eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"])
            subsample = 10 if (data_set == 'test' or lang1 not in eval_lgs) else 5
            n_sentences = 600 if (data_set == 'test' or lang1 not in eval_lgs) else 1500
        elif len(self.params.langs) > 5:
            subsample = 10 if data_set == 'test' else 5
            n_sentences = 300 if data_set == 'test' else 1500
        else:
            # n_sentences = -1 if data_set == 'valid' else 100
            n_sentences = -1
            subsample = 1


        if lang2 is None:
            if self.params.english_only is True:
                n_sentences = 300
            if stream:
                iterator = self.data['mono_stream'][lang1][data_set].get_iterator(shuffle=False, subsample=subsample)
            else:
                iterator = self.data['mono'][lang1][data_set].get_iterator(
                    shuffle=False,
                    group_by_size=True,
                    n_sentences=n_sentences,
                )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data['para'][(_lang1, _lang2)][data_set].get_iterator(
                shuffle=False,
                group_by_size=True,
                n_sentences=n_sentences
            )

        for batch in iterator:
            yield batch if lang2 is None or lang1 < lang2 else batch[::-1]

    def create_reference_files(self):
        """
        Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}

        for (lang1, lang2), v in self.data['para'].items():

            assert lang1 < lang2

            for data_set in ['valid', 'test']:

                # define data paths
                lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_set))
                lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set))

                # store data paths
                params.ref_paths[(lang2, lang1, data_set)] = lang1_path
                params.ref_paths[(lang1, lang2, data_set)] = lang2_path

                # text sentences
                lang1_txt = []
                lang2_txt = []

                # convert to text
                for (sent1, len1), (sent2, len2) in self.get_iterator(data_set, lang1, lang2):
                    lang1_txt.extend(convert_to_text(sent1, len1, self.dico, params))
                    lang2_txt.extend(convert_to_text(sent2, len2, self.dico, params))

                # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                # export hypothesis
                with open(lang1_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang1_txt) + '\n')
                with open(lang2_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang2_txt) + '\n')

                # restore original segmentation
                restore_segmentation(lang1_path)
                restore_segmentation(lang2_path)

    def mask_out(self, x, lengths, rng):
        """
        Decide of random words to mask out.
        We specify the random generator to ensure that the test is the same at each epoch.
        """
        params = self.params
        slen, bs = x.size()

        # words to predict - be sure there is at least one word per sentence
        to_predict = rng.rand(slen, bs) <= params.word_pred
        to_predict[0] = 0
        for i in range(bs):
            to_predict[lengths[i] - 1:, i] = 0
            if not np.any(to_predict[:lengths[i] - 1, i]):
                v = rng.randint(1, lengths[i] - 1)
                to_predict[v, i] = 1
        pred_mask = torch.from_numpy(to_predict.astype(np.uint8))

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        _x_mask = _x_real.clone().fill_(params.mask_index)
        x = x.masked_scatter(pred_mask, _x_mask)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():

            for data_set in ['valid', 'test']:

                # causal prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.clm_steps:
                    self.evaluate_clm(scores, data_set, lang1, lang2)

                # prediction task (evaluate perplexity and accuracy)
                for lang1, lang2 in params.mlm_steps:
                    self.evaluate_mlm(scores, data_set, lang1, lang2)
                
                for lang in params.mass_steps:
                    self.evaluate_mass(scores, data_set, lang)
                
                mass_steps = []
                for lang1 in params.mass_steps:
                    for lang2 in params.mass_steps:
                        if lang1 != lang2:
                            mass_steps.append((lang1, lang2))
                # machine translation task (evaluate perplexity and accuracy)
                for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps] + mass_steps):
                    eval_bleu = params.eval_bleu and params.is_master
                    self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu)

                # report average metrics per language
                _clm_mono = [l1 for (l1, l2) in params.clm_steps if l2 is None]
                if len(_clm_mono) > 0:
                    scores['%s_clm_ppl' % data_set] = np.mean([scores['%s_%s_clm_ppl' % (data_set, lang)] for lang in _clm_mono])
                    scores['%s_clm_acc' % data_set] = np.mean([scores['%s_%s_clm_acc' % (data_set, lang)] for lang in _clm_mono])
                _mlm_mono = [l1 for (l1, l2) in params.mlm_steps if l2 is None]
                if len(_mlm_mono) > 0:
                    scores['%s_mlm_ppl' % data_set] = np.mean([scores['%s_%s_mlm_ppl' % (data_set, lang)] for lang in _mlm_mono])
                    scores['%s_mlm_acc' % data_set] = np.mean([scores['%s_%s_mlm_acc' % (data_set, lang)] for lang in _mlm_mono])

        return scores

    def evaluate_clm(self, scores, data_set, lang1, lang2):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None

        n_words = 0
        xe_loss = 0
        n_valid = 0

        for batch in self.get_iterator(data_set, lang1, lang2, stream=(lang2 is None)):

            # batch
            if lang2 is None:
                x, lengths = batch
                positions = None
                langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
            else:
                (sent1, len1), (sent2, len2) = batch
                x, lengths, positions, langs = concat_batches(sent1, len1, lang1_id, sent2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=True)

            # words to predict
            alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
            pred_mask = alen[:, None] < lengths[None] - 1
            y = x[1:].masked_select(pred_mask[:-1])
            assert pred_mask.sum().item() == y.size(0)

            # cuda
            x, lengths, positions, langs, pred_mask, y = to_cuda(x, lengths, positions, langs, pred_mask, y)

            # forward / loss
            tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=True)
            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_clm_ppl' % (data_set, lang1) if lang2 is None else '%s_%s-%s_clm_ppl' % (data_set, lang1, lang2)
        acc_name = '%s_%s_clm_acc' % (data_set, lang1) if lang2 is None else '%s_%s-%s_clm_acc' % (data_set, lang1, lang2)
        scores[ppl_name] = np.exp(xe_loss / n_words)
        scores[acc_name] = 100. * n_valid / n_words

    def evaluate_mlm(self, scores, data_set, lang1, lang2):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.encoder
        model.eval()
        model = model.module if params.multi_gpu else model

        rng = np.random.RandomState(0)

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None

        n_words = 0
        xe_loss = 0
        n_valid = 0

        for batch in self.get_iterator(data_set, lang1, lang2, stream=(lang2 is None)):

            # batch
            if lang2 is None:
                x, lengths = batch
                positions = None
                langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
            else:
                (sent1, len1), (sent2, len2) = batch
                x, lengths, positions, langs = concat_batches(sent1, len1, lang1_id, sent2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=True)

            # words to predict
            x, y, pred_mask = self.mask_out(x, lengths, rng)

            # cuda
            x, y, pred_mask, lengths, positions, langs = to_cuda(x, y, pred_mask, lengths, positions, langs)

            # forward / loss
            tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += len(y)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_mlm_ppl' % (data_set, lang1) if lang2 is None else '%s_%s-%s_mlm_ppl' % (data_set, lang1, lang2)
        acc_name = '%s_%s_mlm_acc' % (data_set, lang1) if lang2 is None else '%s_%s-%s_mlm_acc' % (data_set, lang1, lang2)
        scores[ppl_name] = np.exp(xe_loss / n_words) if n_words > 0 else 1e9
        scores[acc_name] = 100. * n_valid / n_words if n_words > 0 else 0.


class SingleEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build language model evaluator.
        """
        super().__init__(trainer, data, params)
        self.model = trainer.model


class CombinerEvaluator(Evaluator):
    """ This is an evaluator for word level combiner (not sentence) """

    def __init__(self, trainer, data, params, decoder):

        super().__init__(trainer, data, params)
        self._data = data
        self._params = params
        self._encoder = trainer.encoder
        self._origin_tokens2word = trainer.origin_tokens2word
        self._combiner = trainer.combiner
        self._trained_lang = params.trained_lang
        self._other_lang = params.other_lang
        self._bli = BLI(params.bli_preprocess_method, params.bli_batch_size, params.bli_metric, params.bli_csls_topk)
        self._whole_word_splitter = trainer.whole_word_splitter

        self._whole_word_embedder = SenteceEmbedder(self._encoder, params, self._data["dico"], context_extractor=params.origin_context_extractor)
        self._separated_word_embedder = WordEmbedderWithCombiner(self._encoder, self._combiner, params, self.data["dico"])

        self._loss_function = trainer.loss_function

        # tokenize words if we want to split to word in to char
        self._tokenized_words = read_retokenize_words(params.bped_words_path, self._whole_word_splitter)

        # decoder ( for evaluate word translate)
        self._decoder = decoder

    def evaluate_mt(self, scores, data_set, lang1, lang2, eval_bleu):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs

        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        for batch in self.get_iterator(data_set, lang1, lang2):

            # generate batch
            (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            # encode source sentence
            enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)

            combined_enc, combined_len = combine(enc1, self._combiner, x1, len1, self.dico)

            # decode target sentence
            dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=combined_enc, src_len=combined_len)

            # loss
            word_scores, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

            # generate translation - translate / convert to text
            if eval_bleu:
                max_len = int(1.5 * len1.max().item() + 10)
                if params.beam_size == 1:
                    generated, lengths = decoder.generate(combined_enc, combined_len, lang2_id, max_len=max_len)
                else:
                    generated, lengths = decoder.generate_beam(
                        combined_enc, combined_len, lang2_id, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping,
                        max_len=max_len
                    )
                hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mt_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_mt_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

        # compute BLEU
        if eval_bleu:
            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_set)]

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mt_bleu' % (data_set, lang1, lang2)] = bleu


    def eval_loss(self, scores, lang):
        """
        evaluate combiner valid loss

        """
        params = self._params
        lang_id = params.lang2id[lang]
        data_set = "valid"
        n_words = 0
        all_loss = 0
        for batch, lengths in self.get_iterator(data_set, lang):
            new_batch, new_lengths, whole_word_mask, subword_labels = self.whole_word_splitter.split_batch_sentences(
                batch, lengths, self.data["dico"])

            # original word representations
            self.encoder.eval()
            with torch.no_grad():
                langs =
                origin_word_rep = self.encoder(batch, lengths, langs)
            whole_word_rep = origin_word_rep.masked_select(whole_word_mask)

            # combiner whole word representation
            self.combiner.train()
            combiner_rep = self.combiner(origin_word_rep, new_lengths, subword_labels)

            # mse loss
            loss = self.loss_function(whole_word_rep, combiner_rep)

            n_words += origin_word_rep.size(0)
            all_loss += loss.item() * origin_word_rep.size(0)

        scores["{}_loss".format(data_set)] = all_loss / n_words





class EncDecEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder

    def evaluate_mass(self, scores, data_set, lang):
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang in params.langs

        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        rng = np.random.RandomState(0)

        params = params
        lang_id = params.lang2id[lang]

        n_words = 0
        xe_loss = 0
        n_valid = 0
        for (x1, len1) in self.get_iterator(data_set, lang):
            (x1, len1, x2, len2, y, pred_mask, positions) = self.mask_sent(x1, len1, rng)

            langs1 = x1.clone().fill_(lang_id)
            langs2 = x2.clone().fill_(lang_id)

            # cuda
            x1, len1, langs1, x2, len2, langs2, y, positions = to_cuda(x1, len1, langs1, x2, len2, langs2, y, positions)

            # encode source sentence
            enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)

            enc_mask = x1.ne(params.mask_index)
            enc_mask = enc_mask.transpose(0, 1)
            # decode target sentence
            dec2 = decoder('fwd', x=x2, lengths=len2, 
                           langs=langs2, causal=True, 
                           src_enc=enc1, src_len=len1, positions=positions, enc_mask=enc_mask)
            # loss
            word_scores, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)
            
            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()
            
        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mass_ppl' % (data_set, lang, lang)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_mass_acc' % (data_set, lang, lang)] = 100. * n_valid / n_words

    def mask_sent(self, x, lengths, rng):
        
        def random_start(end):
            p = rng.rand()
            if p >= 0.8:
                return 1
            elif p >= 0.6:
                return end - 1
            else:
                return rng.randint(1, end)

        def mask_word(w):
            p = rng.rand()
            if p >= 0.2:
                return self.params.mask_index
            elif p >= 0.05:
                return rng.randint(self.params.n_words)
            else:
                return w

        positions, inputs, targets, outputs, len2 = [], [], [], [], [] 
        for i in range(lengths.size(0)):
            words = x[:lengths[i], i].tolist()
            l = len(words)
            # Prevent some short sentences will be whole masked
            mask_len = max(1, round(l * self.params.word_mass) - 1)
            start = random_start(l - mask_len + 1)
            len2.append(mask_len)

            pos_i, target_i, output_i, input_i = [], [], [], []
            prev_w = None
            for j, w in enumerate(words):
                if j >= start and j < start + mask_len:
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
        x1 = torch.LongTensor(max(l1) , l1.size(0)).fill_(self.params.pad_index)
        x2 = torch.LongTensor(max(len2), l1.size(0)).fill_(self.params.pad_index)
        y  = torch.LongTensor(max(len2), l1.size(0)).fill_(self.params.pad_index)
        pos = torch.LongTensor(max(len2), l1.size(0)).fill_(self.params.pad_index)
        
        for i in range(l1.size(0)):
            x1[:l1[i], i].copy_(torch.LongTensor(inputs[i]))
            x2[:l2[i], i].copy_(torch.LongTensor(targets[i]))
            y[:l2[i], i].copy_(torch.LongTensor(outputs[i]))
            pos[:l2[i], i].copy_(torch.LongTensor(positions[i]))
        pred_mask = y != self.params.pad_index
        y = y.masked_select(pred_mask)

        return x1, l1, x2, l2, y, pred_mask, pos

    def evaluate_mt(self, scores, data_set, lang1, lang2, eval_bleu):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs

        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        for batch in self.get_iterator(data_set, lang1, lang2):

            # generate batch
            (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            # encode source sentence
            enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)

            # decode target sentence
            dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

            # loss
            word_scores, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()

            # generate translation - translate / convert to text
            if eval_bleu:
                max_len = int(1.5 * len1.max().item() + 10)
                if params.beam_size == 1:
                    generated, lengths = decoder.generate(enc1, len1, lang2_id, max_len=max_len)
                else:
                    generated, lengths = decoder.generate_beam(
                        enc1, len1, lang2_id, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping,
                        max_len=max_len
                    )
                hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mt_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_mt_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

        # compute BLEU
        if eval_bleu:

            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_set)]

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mt_bleu' % (data_set, lang1, lang2)] = bleu


def convert_to_text(batch, lengths, dico, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + '0')
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1


def combine(combiner, encoded, x1, len1, dico):
    """
    combine subword representation to a whole word representation using the combiner
    Params:
        combiner:

        encoded: torch.FloatTensor, shape:[batch_size, max_len, dim]
            encoded representation

        x1: torch.LongTensor, shape: [len, batch_size]

        len1: torch.LongTensor, shape: [batch_size]

        dico: dictionary

    Returns:
        new_encoded: torch.FloatTensor
        new_len: new_length after combine
    """

    x1 = x1.transpose(0, 1)  # [batch_size, len]
    bs, len, _ = x1.size()
    subword_labels = torch.LongTensor(bs, len).fill_(-1)
    for sentence_id, sentence in enumerate(x1):
        cur_subword_labels = []
        subword_finished = True
        for idx in range(len1[sentence_id]):
            cur_token = dico.id2word[sentence[idx].item()]
            if not cur_token.endswith("@@"):
                if subword_finished:
                    subword_labels.append(WHOLEWORD)
                else:
                    subword_labels.append(SUBWORD_END)
                    subword_finished = True
            else:
                subword_finished = False
                subword_labels.append(SUBWORD_FRONT)

    new_encoded, new_lens = combiner.encode(encoded, len1, subword_labels)

    return new_encoded, new_lens
