"""
    evaluate bilingual lexicion
"""
import torch
import pdb
import argparse
import numpy as np
import sys

from src.model.transformer import TransformerModel
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.utils import AttrDict, to_cuda
from tqdm import tqdm
from src.utils import bool_flag
from src.evaluation.get_language_vocab import get_language_vocab
from src.evaluation.bli import eval_bli, translate_words

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

def is_alphabet(uchar):
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
        return True
    else:
        return False

def eval_xlm_bli(path, model_name, dict_path, preprocess, metric, source_vocab=None, target_vocab=None):
    """
    Params:
        path: reload path
        model_name: eval encoder or decoder embedding / or model for xlm
        dict_path:
        preprocess: "ucu"/"u"/..
        metric: "csls", "nn"
        source_vocab: set of source words
        target_voacb: set of target words
    """
    embeddings, dico = load_xlm_embedding(path, model_name)
    if source_vocab is None:
        src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id = unicode_split_chinese_english(embeddings, dico, True) 
    else:
        src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id  = split_language(embeddings, dico, source_vocab, target_vocab)
    scores = eval_bli(
        src_embeddings=src_embeddings,
        tgt_embeddings=tgt_embeddings,
        src_id2word=src_id2word,
        src_word2id=src_word2id,
        tgt_id2word=tgt_id2word,
        tgt_word2id=tgt_word2id,
        dict_path=dict_path,
        preprocess_method=preprocess,
        metric=metric
    )
    return scores

def translate_xlm_chinese(path, model_name, words, preprocess, metric):
    """ use unicode split and translate chinese words , src are chinese ,tgt are english, bpe tokens are also translated"""
    embeddings, dico = load_xlm_embedding(path, model_name)
    src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id = unicode_split_chinese_english(embeddings, dico, drop_bpe=False)
    translate_words(
        src_embeddings=src_embeddings,
        tgt_embeddings=tgt_embeddings,
        src_id2word=src_id2word,
        src_word2id=src_word2id,
        tgt_id2word=tgt_id2word,
        tgt_word2id=tgt_word2id,
        src_words_to_translate=words,
        preprocess_method=preprocess,
        metric=metric,
        print_translation=True
    )


def load_xlm_embedding(path, model_name):
    """
    Load all xlm embeddings
    Params:
        path:
        model_name: model name in the reloaded path, "model" for pretrained xlm encoder; "encoder" for encoder of translation model "decoder" for decoder of translation model
    """
    reloaded = torch.load(path)

    assert model_name in ["model", "encoder", "decoder"]
    state_dict = reloaded[model_name]

    # handle models from multi-GPU checkpoints
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    # reload dictionary and model parameters
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    pretrain_params = AttrDict(reloaded['params'])
    pretrain_params.n_words = len(dico)
    pretrain_params.bos_index = dico.index(BOS_WORD)
    pretrain_params.eos_index = dico.index(EOS_WORD)
    pretrain_params.pad_index = dico.index(PAD_WORD)
    pretrain_params.unk_index = dico.index(UNK_WORD)
    pretrain_params.mask_index = dico.index(MASK_WORD)

    # build model and reload weights
    if model_name != "decoder":
        model = TransformerModel(pretrain_params, dico, True, True)
    else:
        model = TransformerModel(pretrain_params, dico, False, True)
    model.load_state_dict(state_dict)

    return model.embeddings.weight.data, dico

def unicode_split_chinese_english(embeddings, dico, drop_bpe):
    """
    Params:
        embeddings: xlm embeddings
        dico: xlm dictionary
        drop_bpe: don't use tokens which has "@@"
    """
    src_embeddings = []
    tgt_embeddings = []
    src_id2word = {}
    tgt_id2word = {}

    for idx, word in dico.id2word.items():
        if drop_bpe and "@@" in word:
            continue

        if is_chinese(word[0]):
            src_embeddings.append(embeddings[idx])
            src_id2word[len(src_id2word)] = word
        elif is_alphabet(word[0]):
            tgt_embeddings.append(embeddings[idx])
            tgt_id2word[len(tgt_id2word)] = word


    src_word2id = {word:idx for idx, word in src_id2word.items()}
    tgt_word2id = {word:idx for idx, word in tgt_id2word.items()}
    src_embeddings = torch.stack(src_embeddings)
    tgt_embeddings = torch.stack(tgt_embeddings)

    return src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id

def split_language_using_vocab(embeddings, dico, source_vocab, target_vocab):
    """
        Get source and target language based on source vocabulary and target vocabulary
    """
    # check source words are all different from target vocab:
    for source_word in source_vocab:
        assert source_word not in target_vocab, source_word
    for target_word in target_vocab:
        assert target_word not in source_vocab, target_word

    src_embeddings = []
    tgt_embeddings = []
    src_id2word = {}
    tgt_id2word = {}

    for idx, word in dico.id2word.items():
        if word in source_vocab:
            src_embeddings.append(embeddings[idx])
            src_id2word[len(src_id2word)] = word
        elif word in target_vocab:
            tgt_embeddings.append(embeddings[idx])
            tgt_id2word[len(tgt_id2word)] = word


    src_word2id = {word:idx for idx, word in src_id2word.items()}
    tgt_word2id = {word:idx for idx, word in tgt_id2word.items()}
    src_embeddings = torch.stack(src_embeddings)
    tgt_embeddings = torch.stack(tgt_embeddings)

    return src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id

def load_word2vec_embeddings(path, emb_size=-1, drop_bpe=False, vocab=None):
    """ load word2vec/fasttext learned embeddings """
    word2id = {}
    embeddings = []
    with open(path, 'r') as f:
        f.readline() # headline
        for idx, line in enumerate(f):
            if emb_size != -1 and idx >= emb_size:
                break
            if idx % 10000 == 0:
                print("Load {} embeddings".format(idx), file=sys.stderr)
            word, emb = line.rstrip().split(' ', 1)

            # skip some word
            if drop_bpe and "@@" in word:
                continue

            if vocab is not None and word not in vocab:
                continue

            assert word not in word2id
            word2id[word] = len(word2id)
            emb = np.fromstring(emb, sep=' ', dtype=float)
            emb = torch.from_numpy(emb)
            embeddings.append(emb)
    embeddings = torch.stack(embeddings)
    id2word = {idx:word for word, idx in word2id.items()}
    return embeddings, id2word, word2id
            
def read_vocab(vocab_path):
    vocab = set()
    with open(vocab_path, 'r') as f:
        for line in f:
            vocab.add(line.rstrip())
    return vocab


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["xlm", "map"])
    parser.add_argument("--preprocess", default="ucu")
    parser.add_argument("--metric", default="nn")
    parser.add_argument("--dict", help="bilingual dictionary")
    if parser.parse_known_args()[0].method == "xlm":
        parser.add_argument("--reload", help="xlm model path")
        parser.add_argument("--model_name", type=str, choices=["model", "encoder", "decoder"], help='reload a xlm encoder or a xlm translation model')
    elif parser.parse_known_args()[0].method == "map":
        parser.add_argument("--src_embs")
        parser.add_argument("--tgt_embs")
        parser.add_argument("--emb_size", type=int, default=-1)
    parser.add_argument("--using_vocab", type=bool_flag, default=False)
    if parser.parse_known_args()[0].using_vocab is True:
        parser.add_argument("--src_vocab")
        parser.add_argument("--tgt_vocab")

    args = parser.parse_args()

    # maybe load vocab
    if args.using_vocab:
        src_vocab = read_vocab(args.src_vocab)
        tgt_vocab = read_vocab(args.tgt_vocab)
    else:
        src_vocab, tgt_vocab = None, None

    if args.method == "xlm":
        scores = eval_xlm_bli(args.reload, args.model_name, args.dict, args.preporcess, args.metric, src_vocab, tgt_vocab)
        print("Scores: {}".format(scores))
    elif args.method == "map":
        src_embs, src_id2word, src_word2id = load_word2vec_embeddings(args.src_embs, emb_size=args.emb_size, vocab=src_vocab)
        tgt_embs, tgt_id2word, tgt_word2id = load_word2vec_embeddings(args.tgt_embs, emb_size=args.emb_size, vocab=tgt_vocab)
        scores = eval_bli(
            src_embeddings=src_embs,
            tgt_embeddings=tgt_embs,
            src_id2word=src_id2word,
            src_word2id=src_word2id,
            tgt_id2word=tgt_id2word,
            tgt_word2id=tgt_word2id,
            dict_path=args.dict,
            preprocess_method=args.preprocess,
            metric=args.metric
        )
        print("Scores: {}".format(scores))
