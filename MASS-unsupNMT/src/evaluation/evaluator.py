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
from .bli import BLI
from .eval_context_bli import eval_whole_separated_bli, read_retokenize_words, generate_context_word_representation, encode_whole_word_separated_word, generate_and_eval
from src.model.encoder import EncoderInputs
from src.model.seq2seq import DecoderInputs
from src.trainer import mask_sent
from .eval_context_combiner import eval_alignment, AlignmentTypes, AlignmentDataset


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



class Seq2SeqEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.seq2seq_model = trainer.seq2seq_model
        if params.eval_alignment:
            self.alignment_dataset = AlignmentDataset(
                src_bped_path=params.alignment_src_bped_path,
                tgt_bped_path=params.alignment_tgt_bped_path,
                alignment_path=params.alignment_path,
                batch_size=params.batch_size,
                dico=data["dico"],
                src_lang=params.alignment_src_lang,
                tgt_lang=params.alignment_tgt_lang,
                pad_index=params.pad_index,
                eos_index=params.eos_index
            )

    def evaluate_alignment(self, scores):
        type2ave_dis, type2var, type2num = eval_alignment(
            combiner_seq2seq=self.seq2seq_model,
            dataset=self.alignment_dataset,
            lang2id=self.params.lang2id
        )
        for alignment_type in AlignmentTypes:
            score = type2ave_dis[alignment_type]
            scores["alignment-{}-ave-score".format(alignment_type)] = score

    def run_all_evals(self, epoch):
        scores = {}
        scores["epoch"] = epoch

        for data in ["valid", "test"]:
            for lang1, lang2 in self.params.eval_mt_steps:
                self.evaluate_mt(scores, data, lang1, lang2, self.params.eval_bleu and self.params.is_master)

            for lang in self.params.eval_mass_steps:
                self.evaluate_mass(scores, data, lang)

            for lang in self.params.eval_explicit_mass_steps:
                self.evaluate_explicit_mass(scores=scores, data_set=data, lang=lang)

        if self.params.eval_alignment:
            self.evaluate_alignment(scores)

        return scores

    def evaluate_explicit_mass(self, scores, data_set, lang):
        with torch.no_grad():
            params = self.params
            assert data_set in ['valid', 'test']
            assert lang in params.langs

            self.seq2seq_model.eval()
            seq2seq_model = self.seq2seq_model.module if params.multi_gpu else self.seq2seq_model

            rng = np.random.RandomState(0)

            n_words = 0
            xe_loss = 0
            xe_combiner_loss = 0
            n_valid = 0
            n_combiner_words = 0

            for (x1, len1) in self.get_iterator(data_set, lang):
                (x1, len1, x2, len2, y, pred_mask, positions) = mask_sent(x1, len1, rng, self.params, self.data["dico"])
                (x1, len1, x2, len2, y, pred_mask, positions) = to_cuda(x1, len1, x2, len2, y, pred_mask, positions)
                lang_id = self.params.lang2id[lang]
                langs1 = x1.clone().fill_(lang_id)
                langs2 = x2.clone().fill_(lang_id)
                encoder_inputs = EncoderInputs(x1=x1, len1=len1, lang_id=lang_id, langs1=langs1)
                decoder_inputs = DecoderInputs(x2=x2, len2=len2, langs2=langs2, y=y, pred_mask=pred_mask,
                                               positions=positions, lang_id=lang_id)

                word_scores, decoding_loss, combiner_loss, trained_combiner_words = seq2seq_model.explicit_loss(encoder_inputs=encoder_inputs,
                                                                   decoder_inputs=decoder_inputs, get_scores=True)
                # update stats
                n_words += y.size(0)
                xe_loss += decoding_loss.item() * len(y)
                xe_combiner_loss += combiner_loss.item() * trained_combiner_words
                n_combiner_words += trained_combiner_words
                n_valid += (word_scores.max(1)[1] == y.cuda()).long().sum().item()

            # compute perplexity and prediction accuracy
            scores['%s_%s-%s_explicit_mass_ppl' % (data_set, lang, lang)] = np.exp(xe_loss / n_words)
            scores['%s_%s-%s_explicit_mass_acc' % (data_set, lang, lang)] = 100. * n_valid / n_words
            scores["%s-%s-combiner_loss" % (data_set, lang)] = xe_combiner_loss * 1.0 / n_combiner_words

    def evaluate_mass(self, scores, data_set, lang):
        with torch.no_grad():
            params = self.params
            assert data_set in ['valid', 'test']
            assert lang in params.langs

            self.seq2seq_model.eval()
            seq2seq_model = self.seq2seq_model.module if params.multi_gpu else self.seq2seq_model

            rng = np.random.RandomState(0)

            n_words = 0
            xe_loss = 0
            n_valid = 0
            for (x1, len1) in self.get_iterator(data_set, lang):
                (x1, len1, x2, len2, y, pred_mask, positions) = mask_sent(x1, len1, rng, self.params, self.data["dico"])
                (x1, len1, x2, len2, y, pred_mask, positions) = to_cuda(x1, len1, x2, len2, y, pred_mask, positions)
                lang_id = self.params.lang2id[lang]
                langs1 = x1.clone().fill_(lang_id)
                langs2 = x2.clone().fill_(lang_id)
                encoder_inputs = EncoderInputs(x1=x1, len1=len1, lang_id=lang_id, langs1=langs1)
                decoder_inputs = DecoderInputs(x2=x2, len2=len2, langs2=langs2, y=y, pred_mask=pred_mask, positions=positions, lang_id=lang_id)

                word_scores, loss = seq2seq_model.run_seq2seq_loss(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs, get_scores=True)
                # update stats
                n_words += y.size(0)
                xe_loss += loss.item() * len(y)
                n_valid += (word_scores.max(1)[1] == y.cuda()).long().sum().item()

            # compute perplexity and prediction accuracy
            scores['%s_%s-%s_mass_ppl' % (data_set, lang, lang)] = np.exp(xe_loss / n_words)
            scores['%s_%s-%s_mass_acc' % (data_set, lang, lang)] = 100. * n_valid / n_words

    def evaluate_mt(self, scores, data_set, lang1, lang2, eval_bleu):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        with torch.no_grad():
            params = self.params
            assert data_set in ['valid', 'test']
            assert lang1 in params.langs
            assert lang2 in params.langs

            self.seq2seq_model.eval()

            seq2seq_model = self.seq2seq_model.module if params.multi_gpu else self.seq2seq_model

            params = params
            lang1_id = params.lang2id[lang1]
            lang2_id = params.lang2id[lang2]

            # store hypothesis to compute BLEU score
            if eval_bleu:
                hypothesis = []
            n_words = 0
            xe_loss = 0
            n_valid = 0

            for idx, batch in enumerate(self.get_iterator(data_set, lang1, lang2)):
                logger.info("{}".format(idx))

                # generate batch
                (x1, len1), (x2, len2) = batch

                # cuda
                langs1 = x1.clone().fill_(lang1_id)
                langs2 = x2.clone().fill_(lang2_id)

                # target words to predict
                alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
                pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
                y = x2[1:].masked_select(pred_mask[:-1])
                assert len(y) == (len2 - 1).sum().item()

                x1, len1, langs1, x2, len2, langs2, y, pred_mask = to_cuda(x1, len1, langs1, x2, len2, langs2, y, pred_mask)

                encoder_inputs = EncoderInputs(x1=x1, len1=len1, lang_id=lang1_id, langs1=langs1)
                decoder_inputs = DecoderInputs(x2=x2, len2=len2, langs2=langs2, y=y, pred_mask=pred_mask, positions=None, lang_id=lang2_id)

                word_scores, loss, generated, lengths = seq2seq_model.generate_and_run_loss(
                    encoder_inputs=encoder_inputs,
                    decoder_inputs=decoder_inputs,
                    tgt_lang_id=lang2_id,
                    decoding_params=params
                )
                if eval_bleu:
                    hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))

                n_words += y.size(0)
                xe_loss += loss.item() * len(y)
                n_valid += (word_scores.max(1)[1] == y).long().sum().item()

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mt_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_mt_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

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

