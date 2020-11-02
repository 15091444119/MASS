"""
evaluate bilingual combiner
"""

import argparse
import sys
import pdb
from .utils import SenteceEmbedder, WordEmbedderWithCombiner, load_mass_model
from .eval_context_bli import eval_whole_separated_bli, read_retokenize_words, generate_and_eval, BLI, encode_whole_word_separated_word
from src.combiner.combiner import BiLingualCombiner, load_combiner_model
from src.combiner.splitter import WholeWordSplitter


class CombinerBliEvaluator(object):

    def __init__(self, whole_word_embedder, separated_word_embedder, src_splitter, tgt_splitter, src_lang, tgt_lang, eval_params, mass_params, dico, decoder):
        self._dico = dico
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._whole_word_embedder = whole_word_embedder
        self._separated_word_embedder = separated_word_embedder
        self._src_splitter = src_splitter
        self._tgt_splitter = tgt_splitter
        self._src_tokenized_words = read_retokenize_words(eval_params.src_bped_word, src_splitter)
        self._tgt_tokenized_words = read_retokenize_words(eval_params.tgt_bped_word, tgt_splitter)
        self._dict_path = eval_params.dict_path
        self._bli = BLI(eval_params.bli_preprocess_method, eval_params.bli_batch_size, eval_params.bli_metric, eval_params.bli_csls_topk)
        self._save_path = eval_params.save_path
        self._mass_params = mass_params
        self._decoder = decoder

    def run_all_evals(self):
        scores = {}

        src_embs = encode_whole_word_separated_word(self._src_tokenized_words, self._src_lang, self._whole_word_embedder, self._separated_word_embedder)
        tgt_embs = encode_whole_word_separated_word(self._tgt_tokenized_words, self._tgt_lang, self._whole_word_embedder, self._separated_word_embedder)

        self.eval_bli(scores, src_embs, tgt_embs, self._save_path)

        #self.eval_split_whole_word_bli(scores)

        return scores

    def eval_split_whole_word_bli(self, scores):
        """
        Under this setting, we split whole word (with more then one character) using a random bpe helper(so this results is not stable)
        """

        def re_encode_whole_word(words, splitter):
            new_words = []
            for word in words:
                if len(word) == 1:
                    new_words.append(word)
                elif "@@" in word:
                    new_words.append(word)
                else:
                    new_words.append(' '.join(splitter.split_word(word)))
            return new_words

        new_src_bped_words = re_encode_whole_word(self._src_tokenized_words, self._src_splitter)
        new_tgt_bped_words = re_encode_whole_word(self._tgt_tokenized_words, self._tgt_splitter)

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
        all_scores, whole_word_scores, separated_word_scores, sep2whole_scores = eval_whole_separated_bli(
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

        for key, value in sep2whole_scores.items():
            scores["BLI_separate_source2_whole_target " + key] = value

    def eval_encoder_decoder_word_translate(self):

        # generate dictionary, only use source words which can be splitted
        dictionary = {}
        used_srcs = []
        right = 0
        with open(self._dict_path) as f:
            for line in f:
                src, tgt = line.rstrip().split()
                if len(src) <= 1:
                    continue
                src = self._src_splitter.split_word(src)
                if len(src) > 1:
                    src = ' '.join(src)
                    if src not in dictionary:
                        dictionary[src] = [tgt]
                        used_srcs.append(src)
                    else:
                        dictionary[src].append(tgt)
        # translate
        for i in range(0, len(used_srcs), self._mass_params.batch_size):
            words = used_srcs[i: min(len(used_srcs), i + self._mass_params.batch_size)]
            encoded, lengths = self._separated_word_embedder.with_special_token_forward(words, self._src_lang)
            encoded = encoded.transpose(0, 1)
            decoded, dec_lengths = self._decoder.generate(encoded, lengths.cuda(),
                                                          self._mass_params.lang2id[self._tgt_lang],
                                                          max_len=int(1.5 * lengths.max().item() + 10))

            for j in range(decoded.size(1)):
                # remove delimiters
                sent = decoded[:, j]
                delimiters = (sent == self._mass_params.eos_index).nonzero().view(-1)
                assert len(delimiters) >= 1 and delimiters[0].item() == 0
                sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]

                # output translation
                source = used_srcs[i + j]
                dico = self._dico
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


if __name__ == "__main__":

    # moel and data
    parser = argparse.ArgumentParser()
    parser.add_argument("--mass_model", type=str, help="path to mass model")
    parser.add_argument("--src_combiner_model", type=str, help="path to the combiner model")
    parser.add_argument("--tgt_combiner_model", type=str)
    parser.add_argument("--src_bped_word", type=str)
    parser.add_argument("--tgt_bped_word", type=str)
    parser.add_argument("--dict_path", type=str)
    parser.add_argument("--save_path", type=str)


    # bli params
    parser.add_argument("--bli_preprocess_method", type=str, default="u")
    parser.add_argument("--bli_batch_size", type=int, default=64)
    parser.add_argument("--bli_metric", type=str, default="nn")
    parser.add_argument("--bli_csls_topk", type=int, default=10)

    args = parser.parse_args()

    # load models
    dico, mass_params, encoder, decoder = load_mass_model(args.mass_model)
    src_combiner_params, src_combiner = load_combiner_model(args.src_combiner_model)
    tgt_combiner_params, tgt_combiner = load_combiner_model(args.tgt_combiner_model)

    src_lang = src_combiner_params.trained_lang
    tgt_lang = tgt_combiner_params.trained_lang

    assert src_lang != tgt_lang
    assert src_combiner_params.origin_context_extractor == tgt_combiner_params.origin_context_extractor

    # build
    whole_word_embedder = SenteceEmbedder(encoder, mass_params, dico,
                                            context_extractor=src_combiner_params.origin_context_extractor)
    combiner = BiLingualCombiner(src_lang, src_combiner, tgt_lang, tgt_combiner)
    separated_word_embedder = WordEmbedderWithCombiner(encoder, combiner, mass_params, dico)

    src_splitter = WholeWordSplitter.build_splitter(src_combiner_params)
    tgt_splitter = WholeWordSplitter.build_splitter(tgt_combiner_params)

    evaluator = CombinerBliEvaluator(whole_word_embedder=whole_word_embedder,
                                     separated_word_embedder=separated_word_embedder,
                                     src_splitter=src_splitter,
                                     tgt_splitter=tgt_splitter,
                                     src_lang=src_lang,
                                     tgt_lang=tgt_lang,
                                     eval_params=args,
                                     mass_params=src_combiner_params,
                                     dico=dico,
                                     decoder=decoder)

    scores = evaluator.run_all_evals()

    for k, v in scores.items():
        print("{} -> {}".format(k, v))

