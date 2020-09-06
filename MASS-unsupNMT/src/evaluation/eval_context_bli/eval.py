""" evaluate context bli of a give mass model """
import argparse
import torch
import pdb
from ..bli import BLI
from ..utils import  load_mass_model, SenteceEmbedder


def generate_context_word_representation(words, lang, sentence_embedder:SenteceEmbedder, batch_size=128):
    """
    Generate context word representations
    Params:
        words(list): list of strings, bped word, like ["你@@ 好", "我@@ 好"]
        lang: language
        sentence_embedder: a SentenceEmbedder object to generate single representation for each word
        batch_size:

    """
    word2id = {}
    representations = []
    for start_idx in range(0, len(words), batch_size):
        end_idx = min(len(words), start_idx + batch_size)

        # add to word2id
        for idx in range(start_idx, end_idx):
            word = "".join(words[idx]).replace("@@", "")
            assert word not in word2id
            word2id[word] = len(word2id)

        # calculate representation
        with torch.no_grad():
            batch_sentence_representation = sentence_embedder(words[start_idx:end_idx], lang)

        representations.append(batch_sentence_representation)

    representations = torch.cat(representations, dim=0)
    id2word = {idx: word for word, idx in word2id.items()}

    return representations, id2word, word2id


def eval_mass_encoder_context_bli(src_bped_words, src_lang, tgt_bped_words, tgt_lang, sentence_embedder:SenteceEmbedder, bli:BLI):
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
    src_embeddings, src_id2word, src_word2id = generate_context_word_representation(src_bped_words, src_lang, sentence_embedder)
    tgt_embeddings, tgt_id2word, tgt_word2id = generate_context_word_representation(tgt_bped_words, tgt_lang, sentence_embedder)
    scores = bli.eval(src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id)
    return scores


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
    parser.add_argument("--context_extractor", type=str, default="average", choices=["last_time", "average", "max_pool"])

    args = parser.parse_args()

    bli = BLI(args.dict_path, args.preprocess_method, args.batch_size, args.metric, args.csls_topk)

    src_bped_words = read_bped_words(args.src_bped_words_path)
    tgt_bped_words = read_bped_words(args.tgt_bped_words_path)

    dico, mass_params, encoder, _ = load_mass_model(args.model_path)
    sentence_embedder = SenteceEmbedder(encoder, mass_params, dico, args.context_extractor)

    print(eval_mass_encoder_context_bli(src_bped_words, args.src_lang, tgt_bped_words, args.tgt_lang, sentence_embedder, bli))

if __name__ == "__main__":
    main()