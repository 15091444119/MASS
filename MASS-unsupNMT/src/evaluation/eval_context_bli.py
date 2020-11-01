""" evaluate context bli of a give mass model """
import argparse
import torch
import pdb
import sys
import pdb
from .bli import BLI, read_dict
from .utils import SenteceEmbedder, WordEmbedderWithCombiner, load_mass_model
from src.combiner.splitter import WholeWordSplitter


def restore_bpe(tokens):
    """
    Params:
        tokens: string
            bped tokens of a word, tokens are separated by a space
    Returns:
        word: string
            restore bpe
    """

    return tokens.replace("@@", "").replace(" ", "")


class WholeSeparatedEmbs(object):

    def __init__(self, whole_words, separated_word2bpe, word2id, id2word, embeddings):
        self.whole_words = whole_words
        self.separated_word2bpe = separated_word2bpe
        self.word2id = word2id
        self.id2word = id2word
        self.embeddings = embeddings

    def properties(self):
        return self.whole_words, self.separated_word2bpe, self.word2id, self.id2word, self.embeddings

    def separated_words_properties(self):
        """
            In order of original index
            Returns:
                separated_words_id2word:
                separated_words_word2id:
                separated_words_embeddings:
        """
        separated_words_word2id = {}
        separated_words_embeddings = []
        for id in range(len(self.id2word)):
            word = self.id2word[id]
            if word in self.separated_word2bpe:
                separated_words_word2id[word] = len(separated_words_word2id)
                separated_words_embeddings.append(self.embeddings[id])
        separated_words_id2word = {id: word for word, id in separated_words_word2id.items()}
        separated_words_embeddings = torch.stack(separated_words_embeddings, dim=0)

        return separated_words_id2word, separated_words_word2id, separated_words_embeddings

    def whole_words_properties(self):
        """
        In order of original index
        Returns:
            whole_words_id2word:
            whole_words_word2id:
            whole_words_embeddings:
        """
        whole_words_word2id = {}
        whole_words_embeddings = []
        for id in range(len(self.id2word)):
            word = self.id2word[id]
            if word in self.whole_words:
                whole_words_word2id[word] = len(whole_words_word2id)
                whole_words_embeddings.append(self.embeddings[id])
        whole_words_id2word = {id: word for word, id in whole_words_word2id.items()}
        whole_words_embeddings = torch.stack(whole_words_embeddings, dim=0)

        return whole_words_id2word, whole_words_word2id, whole_words_embeddings


def generate_context_word_representation(words, lang, embedder, batch_size=128):
    """
    Generate context word representations
    Params:
        words(list): list of string, bped word, like ["你@@ 好", "我@@ 好"]
        lang: language
        embedder:
        batch_size:
    Returns:
        representation: torch.floattensor, shape=(len(words), emb_dim)
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
        word = restore_bpe(tokens)
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

    if len(whole_words) != 0:
        whole_word_embeddings = generate_context_word_representation(list(whole_words), lang, embedder=whole_word_embedder)
        for word in whole_words:
            word2id[word] = len(word2id)

    if len(separated_word2bpe) != 0:
        separated_word_embeddings = generate_context_word_representation(list(separated_word2bpe.values()), lang, embedder=separated_word_embedder)
        for word in separated_word2bpe:
            word2id[word] = len(word2id)

    id2word = {idx: word for word, idx in word2id.items()}
    if len(whole_words) != 0 and len(separated_word2bpe) != 0:
        embeddings = torch.cat([whole_word_embeddings, separated_word_embeddings], dim=0)
    elif len(whole_words) == 0:
        embeddings = separated_word_embeddings
    elif len(separated_word2bpe) == 0:
        embeddings = whole_word_embeddings

    return WholeSeparatedEmbs(whole_words, separated_word2bpe, word2id, id2word, embeddings)


def generate_and_eval(src_bped_words, src_lang, tgt_bped_words, tgt_lang, dic_path, whole_word_embedder, separated_word_embedder, bli:BLI, save_path=None):
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
    src_whole_separated_embeddings= \
        encode_whole_word_separated_word(bped_words=src_bped_words, lang=src_lang,
                                         whole_word_embedder=whole_word_embedder,
                                         separated_word_embedder=separated_word_embedder)


    tgt_whole_separated_embeddings = \
        encode_whole_word_separated_word(bped_words=tgt_bped_words, lang=tgt_lang,
                                         whole_word_embedder=whole_word_embedder,
                                         separated_word_embedder=separated_word_embedder)

    return eval_whole_separated_bli(src_whole_separated_embeddings, tgt_whole_separated_embeddings, dic_path, bli, save_path)


def eval_whole_separated_bli(src_whole_separated_embeddings: WholeSeparatedEmbs, tgt_whole_separated_embeddings: WholeSeparatedEmbs, dic_path, bli:BLI, save_path=None):
    """
        Evaluate bli on separated whole embeddings
    Params
        src_whole_separated_embeddings: WholeSeparatedEmb
            source embeddings to be evaluated

        tgt_whole_separated_emebddings: WholeSeparatedEmb
            target words to be evaluated

        dic_path: string
            path to load dictionary

        bli: BLI
            A bli method

        save_path: str, default: None
            Path to save bli result, save_path + ".all", save_path + ".whole", save_path + ".separated"
    Returns:
        scores: dict
            top1, top5, top10 bli accuracy
        whole_word_score: dict
        separated_word_score: dict
    """

    # saved_path
    save_all_path = save_path + ".all" if save_path is not None else None
    save_whole_path = save_path + ".whole" if save_path is not None else None
    save_separated_path = save_path + ".separated" if save_path is not None else None

    src_whole_words, src_separated_word2bpe, src_word2id, src_id2word, src_embeddings = src_whole_separated_embeddings.properties()
    tgt_whole_words, tgt_separated_word2bpe, tgt_word2id, tgt_id2word, tgt_embeddings = tgt_whole_separated_embeddings.properties()

    dic = read_dict(dict_path=dic_path, src_word2id=src_word2id, tgt_word2id=tgt_word2id)

    scores = bli.eval(src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id, dic, save_path=save_all_path)

    # get whole source word dictionary and separated word dictionary
    whole_word_dic = {}
    separated_word_dic = {}
    for src_id in dic:
        if src_id2word[src_id] in src_whole_words:
            whole_word_dic[src_id] = dic[src_id]
        else:
            assert src_id2word[src_id] in src_separated_word2bpe
            separated_word_dic[src_id] = dic[src_id]

    whole_word_scores = bli.eval(src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id, whole_word_dic, save_path=save_whole_path)
    separated_word_scores, translation = bli.eval(src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id, separated_word_dic, save_path=save_separated_path, return_translation=True)

    src_evaluated_separated_words = [src_id2word[id] for id in separated_word_dic.keys()]
    # 1. 每一个被切分的词，translate一遍。
    # 2. 每一个被切分的词，找到最近的src whole word
    # 3. 每一个被切分的词，找到最近的src separated word
    # 4. 每一个被切分的词，找到最近的tgt whole word

    src_sep_id2word, src_sep_word2id, src_sep_embs = src_whole_separated_embeddings.separated_words_properties()
    src_whole_id2word, src_whole_word2id, src_whole_embs = src_whole_separated_embeddings.whole_words_properties()
    tgt_whole_id2word, tgt_whole_word2id, tgt_whole_embs = tgt_whole_separated_embeddings.whole_words_properties()
    tgt_sep_id2word, tgt_sep_word2id, tgt_sep_embs = tgt_whole_separated_embeddings.separated_words_properties()

    nearest_tgt_whole = bli.translate_words(src_embeddings, tgt_whole_embs, src_id2word, src_word2id, tgt_whole_id2word, tgt_whole_word2id, src_evaluated_separated_words)
    nearest_tgt_separated = bli.translate_words(src_embeddings, tgt_sep_embs, src_id2word, src_word2id, tgt_sep_id2word, tgt_sep_word2id, src_evaluated_separated_words)
    nearest_src_whole = bli.translate_words(src_embeddings, src_whole_embs, src_id2word, src_word2id, src_whole_id2word, src_whole_word2id, src_evaluated_separated_words)
    nearest_src_separated = bli.translate_words(src_embeddings, src_sep_embs, src_id2word, src_word2id, src_sep_id2word, src_sep_word2id, src_evaluated_separated_words)

    with open(save_path + ".evaluate_sep.txt", 'w') as f:
        for word in src_evaluated_separated_words:
            f.write("Src: {} Target:{}\n".format(word, [tgt_id2word[x] for x in dic[src_word2id[word]]]))
            f.write("Hyp: {}\n".format([tgt_id2word[x] for x in translation[src_word2id[word]]]))
            f.write("Neighbor in tgt whole {}\n".format([tgt_whole_id2word[x] for x in nearest_tgt_whole[src_word2id[word]]]))
            f.write("Neighbor in tgt sep {}\n".format([tgt_sep_id2word[x] for x in nearest_tgt_separated[src_word2id[word]]]))
            f.write("Neighbor in src whole {}\n".format([src_whole_id2word[x] for x in nearest_src_whole[src_word2id[word]]]))
            f.write("Neighbor in src sep {}\n".format([src_sep_id2word[x] for x in nearest_src_separated[src_word2id[word]]]))

    dic2 = read_dict(dic_path, src_sep_word2id, tgt_whole_word2id)



    scores2 = bli.eval(src_sep_embs, tgt_whole_embs, src_sep_id2word, src_sep_word2id, tgt_whole_id2word, tgt_whole_word2id, dic2)
    print("!!")
    print(scores2)
    print("!!")

    return scores, whole_word_scores, separated_word_scores


def read_retokenize_words(path, splitter):
    """
    if splitter is not bpe splitter, retokenize separated word into characters
    """
    words = []
    with open(path, 'r') as f:
        for line in f:
            word = line.rstrip()
            # re split separated word
            if len(word.split()) > 1 and splitter.need_retokenize():
                word = ' '.join(splitter.split_word(restore_bpe(word)))
            words.append(word)
    return words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--src_words_path", help="path to load source words (bped)")
    parser.add_argument("--tgt_words_path", help="path to load target words (bped)")
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--codes_path", type=str, help="bpe codes", default="")
    parser.add_argument("--splitter", type=str, help="splitter, bpe or char", choices=["BPE", "CHAR", "ROB"])

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

    splitter = WholeWordSplitter.build_splitter(args)
    src_bped_words = read_retokenize_words(args.src_words_path, splitter=splitter)
    tgt_bped_words = read_retokenize_words(args.tgt_words_path, splitter=splitter)

    dico, mass_params, encoder, _ = load_mass_model(args.model_path)
    sentence_embedder = SenteceEmbedder(encoder, mass_params, dico, args.context_extractor)

    scores, whole_word_scores, separated_word_scores = generate_and_eval(
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