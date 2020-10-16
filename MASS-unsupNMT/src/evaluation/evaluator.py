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
        self._model = trainer.model
        self._origin_tokens2word = trainer.origin_tokens2word
        self._combiner = trainer.combiner
        self._whole_word_embedder = SenteceEmbedder(trainer.model, params, data["dico"], context_extractor=params.origin_context_extractor)
        self._separated_word_embedder = WordEmbedderWithCombiner(trainer.model, trainer.combiner, params, data["dico"])

        # used to evaluate none parameter word embedder
        self._non_para_word_embedder = SenteceEmbedder(trainer.model, params, data["dico"], context_extractor=params.combiner_context_extractor)

        self._src_lang = params.dict_src_lang
        self._tgt_lang = params.dict_tgt_lang
        self._bli = BLI(params.bli_preprocess_method, params.bli_batch_size, params.bli_metric, params.bli_csls_topk)
        self._whole_word_splitter = trainer.whole_word_splitter
        self._loss_function = trainer.loss_function
        self._src_tokenized_words = read_retokenize_words(params.src_bped_words_path, self._whole_word_splitter)
        self._tgt_tokenized_words = read_retokenize_words(params.tgt_bped_words_path, self._whole_word_splitter)

        # decoder ( for evaluate word translate)
        self._decoder = decoder

    def eval_encoder_decoder_word_translate(self):

        # generate dictionary, only use source words which can be splitted
        dictionary = {}
        used_srcs = []
        right = 0
        with open(self._params.dict_path) as f:
            for line in f:
                src, tgt = line.rstrip().split()
                if len(src) <= 1:
                    continue
                src = self._whole_word_splitter.split_word(src)
                if len(src) > 1:
                    src = ' '.join(src)
                    if src not in dictionary:
                        dictionary[src] = [tgt]
                        used_srcs.append(src)
                    else:
                        dictionary[src].append(tgt)

        # translate
        for i in range(0, len(used_srcs), self._params.batch_size):
            words = used_srcs[i: min(len(used_srcs), i + self._params.batch_size)]
            encoded, lengths = self._separated_word_embedder.with_special_token_forward(words, self._src_lang)
            encoded = encoded.transpose(0, 1)
            decoded, dec_lengths = self._decoder.generate(encoded, lengths.cuda(), self._params.lang2id[self._tgt_lang], max_len=int(1.5 * lengths.max().item() + 10))

            for j in range(decoded.size(1)):
                # remove delimiters
                sent = decoded[:, j]
                delimiters = (sent == self._params.eos_index).nonzero().view(-1)
                assert len(delimiters) >= 1 and delimiters[0].item() == 0
                sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]

                # output translation
                source = used_srcs[i + j]
                dico = self._data["dico"]
                target = " ".join([dico[sent[k].item()] for k in range(len(sent))])
                if target.replace("@@", "").replace(" ", ' ') in dictionary[source]:
                    ans = "True"
                else:
                    ans = "False"
                if ans == "True":
                    right += 1
                sys.stderr.write("%i %s -> %s\nans:%s\n" % (i + j, source, target, ans))
                pdb.set_trace()

        sys.stderr.write("acc:{}".format(right / len(dictionary)))


    def eval_non_para(self):
        # small hack here to run all evals on none parameter embedder
        # TODO loss here is not right, maybe we should not use sentence embedder or word embedderwith combiner
        tmp_separated = self._separated_word_embedder
        tmp_whole = self._whole_word_embedder

        self._separated_word_embedder = self._non_para_word_embedder
        self._whole_word_embedder = self._whole_word_embedder
        scores = self.run_all_evals(-1)

        self._separated_word_embedder = tmp_separated
        self._whole_word_embedder = tmp_whole

        return scores

    def run_all_evals(self, epoch):
        """
        Rewrite parent method
        """
        scores = OrderedDict({'epoch': epoch})

        # evaluate combiner
        for lang in self.params.combiner_steps:
            self.eval_loss(scores, lang)
        scores["valid-average-loss"] = sum(
            [scores["valid-{}-combiner".format(lang)] for lang in self.params.combiner_steps]) / 2.0

        all_embs = {}
        # this is calculated only once to save time
        all_embs["src"] = encode_whole_word_separated_word(self._src_tokenized_words, self._src_lang, self._whole_word_embedder, self._separated_word_embedder)
        all_embs["tgt"] = encode_whole_word_separated_word(self._tgt_tokenized_words, self._tgt_lang, self._whole_word_embedder, self._separated_word_embedder)
        for lang in ["src", "tgt"]:
            for data in ["valid", "train"]:
                self.eval_combiner_acc(scores, data, lang, all_embs[lang], save_path=os.path.join(self.params.dump_path, "{}_{}_{}".format(epoch, lang, data)))

        # evaluate bli
        self.eval_bli(scores, all_embs["src"], all_embs["tgt"], save_path=os.path.join(self.params.dump_path, "{}_bli".format(epoch)))
            #self.eval_split_whole_word_bli(scores)

        return scores

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
            new_batch, new_lengths= self._whole_word_splitter.re_encode_batch_words(batch, lengths,
                                                                                                self._data["dico"],
                                                                                                params)

            batch, lengths, new_batch, new_lengths = to_cuda(batch, lengths, new_batch, new_lengths)

            langs = batch.clone().fill_(lang_id)
            self._model.eval()
            self._combiner.eval()
            with torch.no_grad():
                # original encode
                origin_encoded = self._model('fwd', x=batch, lengths=lengths, langs=langs, causal=False)
                origin_word_rep = self._origin_tokens2word(origin_encoded.transpose(0, 1), lengths)

                # new encode
                langs = new_batch.clone().fill_(lang_id)
                new_encoded = self._model('fwd', x=new_batch, lengths=new_lengths, langs=langs, causal=False)
                new_word_rep = self._combiner(new_encoded, new_lengths, lang)

            # mse loss
            loss = self._loss_function(origin_word_rep, new_word_rep)

            n_words += origin_word_rep.size(0)
            all_loss += loss.item() * origin_word_rep.size(0)

        scores["{}-{}-combiner".format(data_set, lang)] = all_loss / n_words

    def check_dataset(self):
        """
        Assert all data are whole word and valid set have no intersection with training set
        """
        for lang in self.params.combiner_steps:
            valid_word_idxs = set()
            for batch, lengths in self.get_iterator("valid", lang):
                assert (lengths == 3).all()
                for word in batch.transpose(0, 1):
                    valid_word_idxs.add(word[1].item())
            for batch, lengths in self.get_iterator("train", lang):
                assert (lengths == 3).all()
                for word in batch.transpose(0, 1):
                    assert word[1].item() not in valid_word_idxs, self._data["dico"].id2word[word[1].item()]
        logger.info("Training data and valid data have no intersection.")

    def eval_split_whole_word_bli(self, scores):
        """
        Under this setting, we split whole word (with more then one character) using a random bpe helper(so this results is not stable)
        """

        def re_encode_whole_word(words):
            new_words = []
            for word in words:
                if len(word) == 1:
                    new_words.append(word)
                elif "@@" in word:
                    new_words.append(word)
                else:
                    new_words.append(' '.join(self._whole_word_splitter.split_word(word)))
            return new_words

        new_src_bped_words = re_encode_whole_word(self._src_tokenized_words)
        new_tgt_bped_words = re_encode_whole_word(self._tgt_tokenized_words)

        all_scores, whole_word_scores, separated_word_scores = generate_and_eval(
            src_bped_words=new_src_bped_words,
            src_lang=self._src_lang,
            tgt_bped_words=new_tgt_bped_words,
            tgt_lang=self._tgt_lang,
            dic_path=self._params.dict_path,
            whole_word_embedder=self._separated_word_embedder,
            separated_word_embedder=self._separated_word_embedder,
            bli=self._bli,
            save_path=None)

        for key, value in all_scores.items():
            scores["BLI_split_all " + key] = value

        for key, value in whole_word_scores.items():
            scores["BLI_split_whole_word " + key] = value

        for key, value in separated_word_scores.items():
            scores["BLI_split_separated_word " + key] = value

    def eval_bli(self, scores, src_whole_separated_embeddings, tgt_whole_separated_embeddings, save_path=None):
        all_scores, whole_word_scores, separated_word_scores = eval_whole_separated_bli(
            src_whole_separated_embeddings=src_whole_separated_embeddings,
            tgt_whole_separated_embeddings=tgt_whole_separated_embeddings,
            dic_path=self._params.dict_path,
            bli=self._bli,
            save_path=save_path)

        for key, value in all_scores.items():
            scores["BLI_all " + key] = value

        for key, value in whole_word_scores.items():
            scores["BLI_whole_word " + key] = value

        for key, value in separated_word_scores.items():
            scores["BLI_separated_word " + key] = value

    def eval_combiner_acc(self, scores, data, src_or_tgt, whole_separated_embeddings, save_path=None):
        """
            For each word in the valid set and training set(this are whole word), we first split into bpe tokens,
        then get the combiner representation of it, then we search nearest neighbor in the original embedding
        space(given by src_bped_words or tgt_bped_words) and see if the nearest neighbor is that word.
            This can be regarded as a BLI from combiner space to original mass output space.
            The original mass output space may contain whole words and separated words,
            So two original space will be considerd:
                1. whole words
                2. whole words + combined separated words

        Example:
            你好 -> 你@@ 好 -> representation([你@@, 好]) -> search nearest neighbor in (你好，我好，大家好，...）

        Params:
            scores: dict

            data: string, choices=["valid", "train"]

            src_or_tgt: string, choices=["src", "tgt"]

            whole_separated_embeddings: WholeSeparatedEmbs
                The original embedding space

            save_path: str
                Path to save nearest neighbor of each word in the valid set
        """
        if src_or_tgt == "src":
            lang = self._src_lang
        else:
            lang = self._tgt_lang

        # generate combiner space representation
        combiner_word2id = {}
        combiner_words = []  # re bped combiner words for generate representation
        for batch, length in self.get_iterator(data, lang):
            assert (length == 3).all() # all are words
            batch = batch.transpose(0, 1)
            for token_idxs in batch:
                word_id = token_idxs[1].item()  # [eos, word_id, eos]
                word = self._data["dico"].id2word[word_id]
                combiner_word2id[word] = len(combiner_word2id)
                re_bped_word = ' '.join(self._whole_word_splitter.split_word(word))
                combiner_words.append(re_bped_word)
        assert len(combiner_word2id) == len(combiner_words)
        combiner_embeddings = generate_context_word_representation(combiner_words, lang, self._separated_word_embedder)
        combiner_id2word = {idx: word for word, idx in combiner_word2id.items()}


        # generate original mass representation
        whole_words, separated_words2bpe, origin_word2id, origin_id2word, origin_embeddings = whole_separated_embeddings.properties()

        # generate a dictionary
        dic = {}
        for word, combiner_idx in combiner_word2id.items():
            assert word in origin_word2id
            origin_idx = origin_word2id[word]
            dic[combiner_idx] = [origin_idx]

        logger.info("Number of combiner word: {} Number of origin word: {} Number of dic word: {}".format(len(combiner_id2word), len(origin_word2id), len(dic)))

        # bli on whole + separated space
        whole_separated_save_path = save_path + "_whole_sepa" if save_path is not None else None
        bli_scores = self._bli.eval(combiner_embeddings, origin_embeddings, combiner_id2word, combiner_word2id, origin_id2word, origin_word2id, dic, save_path=whole_separated_save_path)
        for key, value in bli_scores.items():
            scores["{data}-{lang}-whole-sepa-combiner-acc-{key}".format(data=data, lang=lang, key=key)] = value

        # only whole words space
        whole_id2word, whole_word2id, whole_embeddings = whole_separated_embeddings.whole_words_properties()
        dic = {}
        for word, combiner_idx in combiner_word2id.items():
            assert word in whole_word2id
            whole_idx = whole_word2id[word]
            dic[combiner_idx] = [whole_idx]
        whole_save_path = save_path +"_whole" if save_path is not None else None
        whole_bli_scores = self._bli.eval(combiner_embeddings, whole_embeddings, combiner_id2word, combiner_word2id, whole_id2word, whole_word2id, dic, save_path=whole_save_path)
        for key, value in whole_bli_scores.items():
            scores["{data}-{lang}-whole-combiner-acc-{key}".format(data=data, lang=lang, key=key)] = value



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
