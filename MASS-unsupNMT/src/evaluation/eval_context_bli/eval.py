""" evaluate context bli of a give mass model """
import argparse
import torch
import sys
import pdb
from ..bli import BLI, read_dict
from ..utils import  load_mass_model, SenteceEmbedder


def generate_context_word_representation(words, lang, sentence_embedder:SenteceEmbedder, batch_size=128):
    """
    Generate context word representations
    Params:
        words(list): list of string, bped word, like ["你@@ 好", "我@@ 好"]
        lang: language
        sentence_embedder: a SentenceEmbedder object to generate single representation for each word
        batch_size:

    """
    word2id = {}
    representations = []
    last_print = 0
    for start_idx in range(0, len(words), batch_size):
        end_idx = min(len(words), start_idx + batch_size)

        # add to word2id
        batch = []
        for idx in range(start_idx, end_idx):
            word = words[idx].replace("@@", "").replace(" ", "")
            if word in word2id:
                continue
            word2id[word] = len(word2id)
            batch.append(words[idx])

        # calculate representation
        with torch.no_grad():
            batch_sentence_representation = sentence_embedder(batch, lang)

        representations.append(batch_sentence_representation)
        if (start_idx - last_print >= 10000):
            print(start_idx, file=sys.stderr)
            last_print = start_idx

    representations = torch.cat(representations, dim=0)
    id2word = {idx: word for word, idx in word2id.items()}

    return representations, id2word, word2id


def eval_mass_encoder_context_bli(src_bped_words, src_lang, tgt_bped_words, tgt_lang, dic_path, sentence_embedder:SenteceEmbedder, bli:BLI, save_path=None):
    """
        1. Generate context representation for each source word and each target word
        2. evaluate bli on it
    Params
        src_bped_words: words to be evaluated (bped using the same code as the mass model)
        tgt_bped_words:
        src_lang: language of source word
        tgt_lang: language of target word
        sentence_embedder:
        bli:
    Returns:
        scores(dict): top1, top5, top10 bli accuracy
    """
    src_word2length = {tokens.replace("@@", "").replace(" ", ""): len(tokens.split()) for tokens in src_bped_words}
    tgt_word2length = {tokens.replace("@@", "").replace(" ", ""): len(tokens.split()) for tokens in tgt_bped_words}
    src_embeddings, src_id2word, src_word2id = generate_context_word_representation(src_bped_words, src_lang, sentence_embedder)
    tgt_embeddings, tgt_id2word, tgt_word2id = generate_context_word_representation(tgt_bped_words, tgt_lang, sentence_embedder)
    dic = read_dict(dict_path=dic_path, src_word2id=src_word2id, tgt_word2id=tgt_word2id)

    scores = bli.eval(src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id, dic, save_path=save_path)

    whole_word_dic = {}
    seperated_word_dic = {}
    for src_id in dic:
        if src_word2length[src_id2word[src_id]] == 1:
            whole_word_dic[src_id] = dic[src_id]
        else:
            seperated_word_dic[src_id] = dic[src_id]

    whole_word_scores = bli.eval(src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id, whole_word_dic)
    seperated_word_scores = bli.eval(src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id, seperated_word_dic)

    return scores, whole_word_scores, seperated_word_scores


def read_bped_words(path):
    words = []
    with open(path, 'r') as f:
        for line in f:
            words.append(line.rstrip())
    return words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--src_bped_words_path", help="path to load source words (bped)")
    parser.add_argument("--tgt_bped_words_path", help="path to load target words (bped)")
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)

    # bli params
    parser.add_argument("--preprocess_method", type=str, default="ucu")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--csls_topk", type=int, default=100)
    parser.add_argument("--metric", type=str, default="nn")
    parser.add_argument("--dict_path", type=str)
    parser.add_argument("--context_extractor", type=str, default="average", choices=["last_time", "average", "max_pool", "before_eos"])
    parser.add_argument("--save_path", type=str, default=None, help="path to save bli result")

    args = parser.parse_args()

    bli = BLI(args.preprocess_method, args.batch_size, args.metric, args.csls_topk)

    src_bped_words = read_bped_words(args.src_bped_words_path)
    tgt_bped_words = read_bped_words(args.tgt_bped_words_path)

    dico, mass_params, encoder, _ = load_mass_model(args.model_path)
    sentence_embedder = SenteceEmbedder(encoder, mass_params, dico, args.context_extractor)

    scores, whole_word_scores, seperated_word_scores = eval_mass_encoder_context_bli(src_bped_words, args.src_lang, tgt_bped_words, args.tgt_lang, args.dict_path, sentence_embedder, bli)
    print("scores: {}\nwhole word scores{}\nseperated word scores{}\n".format(scores, whole_word_scores, seperated_word_scores))

if __name__ == "__main__":
    main()
