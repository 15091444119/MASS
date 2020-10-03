""" evaluate context bli of a give mass model """
import argparse
import torch
import pdb
import sys
import pdb
from .bli import BLI, read_dict
from .utils import SenteceEmbedder, WordEmbedderWithCombiner, load_mass_model


from sklearn import cluster
def generate_context_word_representation(words, lang, embedder, batch_size=128):
    """
    Generate context word representations
    Params:
        words(list): list of string, bped word, like ["你@@ 好", "我@@ 好"]
        lang: language
        embedder:
        batch_size:
    """
    representations = []
    last_print = 0
    embedder.eval()
    for start_idx in range(0, len(words), batch_size):
        end_idx = min(len(words), start_idx + batch_size)

        # add to word2id
        batch = words[start_idx:end_idx]

        # calculate representation
        with torch.no_grad():
            batch_sentence_representation = embedder(batch, lang)

        representations.append(batch_sentence_representation)
        if (start_idx - last_print >= 10000):
            print(start_idx, file=sys.stderr)
            last_print = start_idx

    representations = torch.cat(representations, dim=0)

    return representations


def split_whole_separate(bped_words):
    """
    split bped words into two groups: whole words and separated words
    params:
        bped_words: list of strings
            each string is a word, maybe tokenized by bpe, like "你@@ 好"
    returns:
        single_words: set
            set of single_words

        separated_words2bpe: dict
            key: separated word, value: it's bpe format
            example: {"你好": "你@@ 好"}
    """
    whole_words = set()
    separated_word2bpe = {}

    for tokens in bped_words:
        word = tokens.replace("@@", "").replace(" ", "")
        lengths = len(tokens.split())

        if word in whole_words or word in separated_word2bpe:
            continue

        if lengths == 1:
            whole_words.add(word)
        elif lengths > 1:
            separated_word2bpe[word] = tokens

    return whole_words, separated_word2bpe


def encode_whole_word_separated_word(bped_words, lang, whole_word_embedder, separated_word_embedder):
    """
    Get representation for single word and bpe separated word

    Params:
        bped_words: list of string
            Words to encode

        lang: string
            Language

        whole_word_embedder:
            Embedder to encode whole words

        separated_word_embedder:
            Embedder to encode separated words

    Returns:
        whole_words: set

        separated_word2bpe: dict
            like: {"你好": "你@@ 好"}

        word2id: dict

        id2word: dict

        embeddings: torch.FloatTensor, shape(number of words, hidden_dim)
            the encoded embeddings for each word, in the order of id
    """
    whole_words, separated_word2bpe = split_whole_separate(bped_words)
    word2id = {}

    whole_word_embeddings = generate_context_word_representation(list(whole_words), lang, embedder=whole_word_embedder)
    for word in whole_words:
        word2id[word] = len(word2id)

    separated_word_embeddings = generate_context_word_representation(list(separated_word2bpe.values()), lang, embedder=separated_word_embedder)
    for word in separated_word2bpe:
        word2id[word] = len(word2id)

    id2word = {idx: word for word, idx in word2id.items()}

    embeddings = torch.cat([whole_word_embeddings, separated_word_embeddings], dim=0)

    return whole_words, separated_word2bpe, word2id, id2word, embeddings


def eval_context_bli(src_bped_words, src_lang, tgt_bped_words, tgt_lang, dic_path, whole_word_embedder, separated_word_embedder:WordEmbedderWithCombiner, bli:BLI, save_path=None):
    """
        1. Generate context representation for each source word and each target word
        2. evaluate bli on it
    Params
        src_bped_words: list of strings
            source words to be evaluated (bped using the same code as the mass model)

        tgt_bped_words:
            words to be evaluated (bped using the same code as the mass model)

        src_lang: str
            language of source word

        tgt_lang: str
            language of target word

        whole_word_embedder: SentenceEmbedder or WordEmbedderWithCombiner
            Embedder used to get the embedding of whole words

        separated_word_embedder: WordEmbedderWithCombiner
            Embedder used to get the embedding of separated words

        bli: BLI
            A bli method

        save_path: str, default: None
            Path to save bli result
    Returns:
        scores: dict
            top1, top5, top10 bli accuracy
        whole_word_score: dict
        separated_word_score: dict
    """
    src_whole_words, src_separated_word2bpe, src_word2id, src_id2word, src_embeddings = \
        encode_whole_word_separated_word(bped_words=src_bped_words, lang=src_lang, whole_word_embedder=whole_word_embedder, separated_word_embedder=separated_word_embedder)
    tgt_whole_words, tgt_separated_word2bpe, tgt_word2id, tgt_id2word, tgt_embeddings = \
        encode_whole_word_separated_word(bped_words=tgt_bped_words, lang=tgt_lang, whole_word_embedder=whole_word_embedder, separated_word_embedder=separated_word_embedder)

    dic = read_dict(dict_path=dic_path, src_word2id=src_word2id, tgt_word2id=tgt_word2id)

    scores = bli.eval(src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id, dic, save_path=save_path)

    # get whole source word dictionary and separated word dictionary
    whole_word_dic = {}
    seperated_word_dic = {}
    for src_id in dic:
        if src_id2word[src_id] in src_whole_words:
            whole_word_dic[src_id] = dic[src_id]
        else:
            assert src_id2word[src_id] in src_separated_word2bpe
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

    scores, whole_word_scores, separated_word_scores = eval_context_bli(
        src_bped_words=src_bped_words,
        src_lang=args.src_lang,
        tgt_bped_words=tgt_bped_words,
        tgt_lang=args.tgt_lang,
        dic_path=args.dict_path,
        whole_word_embedder=sentence_embedder,
        separated_word_embedder=sentence_embedder,
        bli=bli,
        save_path=args.save_path)
    print("scores: {}\nwhole word scores{}\nseparated word scores{}\n".format(scores, whole_word_scores, separated_word_scores))

if __name__ == "__main__":
    main()