"""
evaluate bilingual combiner
"""

from .utils import SenteceEmbedder, WordEmbedderWithCombiner
from .eval_context_bli import eval_whole_separated_bli, read_retokenize_words, generate_and_eval, BLI


class CombinerBliEvaluator(object):

    def __init__(self, encoder, src_combiner, tgt_combiner, splitter, dico, params):
        self._dico = dico
        self._whole_word_embedder = SenteceEmbedder(encoder, params, dico, context_extractor=params.origin_context_extractor)

        self._separated_word_embedder = WordEmbedderWithCombiner(encoder, src_combiner, params, dico)
        self._splitter = splitter
        self._src_lang = params.src_lang
        self._tgt_lang = params.tgt_lang
        self._src_tokenized_words = read_retokenize_words(params.src_bped_path, self._splitter)
        self._tgt_tokenized_words = read_retokenize_words(params.tgt_bped_path, self._splitter)
        self._dict_path = params.dict_path
        self._bli = BLI(params.bli_preprocess_method, params.bli_batch_size, params.bli_metric, params.bli_csls_topk)

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
            dic_path=self._dict_path,
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
            dic_path=self._dict_path,
            bli=self._bli,
            save_path=save_path)

        for key, value in all_scores.items():
            scores["BLI_all " + key] = value

        for key, value in whole_word_scores.items():
            scores["BLI_whole_word " + key] = value

        for key, value in separated_word_scores.items():
            scores["BLI_separated_word " + key] = value