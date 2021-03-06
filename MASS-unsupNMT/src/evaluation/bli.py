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

def topk_mean(src_embs, tgt_embs, batch_size, csls_topk):
    """
    mean of the topk cos distance(used for csls)
    """
    knn_cos_mean = torch.FloatTensor(len(tgt_embs))
    if torch.cuda.is_available():
        knn_cos_mean = knn_cos_mean.cuda()
    for i in range(0, len(tgt_embs), batch_size):
        j = min(len(tgt_embs), i + batch_size)
        simi = torch.matmul(tgt_embs[i:j], src_embs.transpose(0, 1))
        simi, idxs = torch.topk(simi, csls_topk, dim=-1)
        knn_cos_mean[i:j] = simi.mean(dim=-1)
    return knn_cos_mean

def calculate_word_translation_accuracy(translation, truth_dict, topk):
    """ get topk accuarcy
    params:
        translation:
        truth_dict:
        topk: use topk translation
    example:
        translation = {1:[3, 2, 4]}
        truth_dict = {1:[2, 5]}
        topk = 2
        -> calculate_word_translation_accuracy(translation, truth_dict, topk) = 1.0
    """

    # assert all words are translated
    for src in truth_dict:
        assert src in translation
    assert len(translation) == len(truth_dict)

    result = []  
    for src, tgt_list in translation.items():
        have_translation = False
        for idx, tgt in enumerate(tgt_list):
            if idx == topk:
                break
            if tgt in truth_dict[src]:
                have_translation = True
                break
        result.append(have_translation)
    
    acc = result.count(True) / len(result)

    return acc


def retrieval(src_mapped, tgt_mapped, src_ids, csls_topk, batch_size, metric, num_neighbor=10):
    """ get  nearest neighbors as translation for word in src_ids
        src_ids: src_ids to get translation
    """
    if metric == "csls":
        # CSLS
        translation = {}
        knn_cos_mean = topk_mean(src_mapped, tgt_mapped, csls_topk=csls_topk, batch_size=batch_size)
        for i in range(0, len(src_ids), batch_size):
            j = min(len(src_ids), i + batch_size)
            simi = torch.matmul(src_mapped[src_ids[i:j]], tgt_mapped.transpose(0, 1))
            csls_score = 2 * simi - knn_cos_mean
            _, indices = csls_score.topk(k=num_neighbor, dim=-1)
            for k in range(j - i):
                translation[src_ids[i + k]] = [indice.item() for indice in indices[k]]
    elif metric == "nn":
        # NN
        translation = {}
        for i in range(0, len(src_ids), batch_size):
            j = min(len(src_ids), i + batch_size)
            simi = torch.matmul(src_mapped[src_ids[i:j]], tgt_mapped.transpose(0, 1))
            _, indices = simi.topk(k=num_neighbor, dim=-1)
            for k in range(j - i):
                translation[src_ids[i + k]] = [indice.item() for indice in indices[k]]
    return translation

def preprocess_embedding(emb, preprocess_method):
    for char in preprocess_method:
        if char == "u":  # length normalize
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        elif char == "c":  # mean center
            emb = emb - emb.mean(dim=0)
        else:
            raise ValueError
    return emb

def read_dict(dict_path, src_word2id, tgt_word2id):
    dic = {}
    dropped_pair_count = 0
    with open(dict_path, 'r') as f:
        for line in f:
            src, tgt = line.rstrip().split()
            if src in src_word2id and tgt in tgt_word2id:
                src_id = src_word2id[src]
                if src_id in dic:
                    dic[src_id].append(tgt_word2id[tgt])
                else:
                    dic[src_id] = [tgt_word2id[tgt]]
            else:
                dropped_pair_count += 1
    print("Dropped {} pairs, resulting in dictionary of {} source words".format(dropped_pair_count, len(dic)), file=sys.stderr)
    return dic

def eval_bli(
    src_embeddings,
    tgt_embeddings,
    src_id2word,
    src_word2id,
    tgt_id2word,
    tgt_word2id,
    dict_path,
    preprocess_method="",
    batch_size=500,
    metric="nn",
    csls_topk=10):
    """ evaluate bilingual lexicion induction """

    if torch.cuda.is_available():
        src_embeddings = src_embeddings.cuda()
        tgt_embeddings = tgt_embeddings.cuda()

    if preprocess_method != "":
        src_embeddings = preprocess_embedding(src_embeddings, preprocess_method)
        tgt_embeddings = preprocess_embedding(tgt_embeddings, preprocess_method)
    
    dic = read_dict(dict_path, src_word2id, tgt_word2id)

    translation = retrieval(src_embeddings, tgt_embeddings, list(dic.keys()), csls_topk=csls_topk, batch_size=batch_size, metric=metric)


    top1_acc = calculate_word_translation_accuracy(translation, dic, topk=1)
    top5_acc = calculate_word_translation_accuracy(translation, dic, topk=5) 
    top10_acc = calculate_word_translation_accuracy(translation, dic, topk=10) 
    
    scores = {"top1_acc":top1_acc, "top5_acc":top5_acc, "top10_acc":top10_acc}

    return scores

def eval_xlm_bli(embeddings, dico, dict_path, preprocess, metric, use_language_suffix=False):
    """
    1. extract source and target embedding from the mixed xlm embedding
    2. evaluate dictionary induction on extracted source and target embeddings

    Params:
        embeddings: embeddings of XLM
        dico: dictionary object of the XLM
        dict_path: path to load dictionary
        use_language_suffix: use language suffix like [LANG=zh] [LANG=en] to detect language
    """
    src_embeddings = []
    tgt_embeddings = []
    src_id2word = {}
    tgt_id2word = {}

    # extract 
    for idx, word in dico.id2word.items():
        
        # escape bpe word piece
        if "@@" in word:
            continue

        # detect langauge 
        if is_chinese(word[0]):
            if use_language_suffix and not  word.endswith("[LANG=zh]"):
                continue
            src_embeddings.append(embeddings[idx])
            src_id2word[len(src_id2word)] = word
        elif is_alphabet(word[0]):
            if use_language_suffix and not word.endswith("[LANG=en]"):
                continue
            tgt_embeddings.append(embeddings[idx])
            tgt_id2word[len(tgt_id2word)] = word
        else:
            #print("Dropped :{}".format(word))
            pass
    
    print("Extract {} chinese word, {} english word".format(len(src_id2word), len(tgt_id2word)), file=sys.stderr)
    src_word2id = {word:idx for idx, word in src_id2word.items()}
    tgt_word2id = {word:idx for idx, word in tgt_id2word.items()}
    src_embeddings = torch.stack(src_embeddings)
    tgt_embeddings = torch.stack(tgt_embeddings)
   
    """
    while True:
        print("input word1")
        word1 = input()
        print("input word2")
        word2 = input()
        if word1 in src_word2id:
            emb1 = src_embeddings[src_word2id[word1]]
        elif word1 in tgt_word2id:
            emb1 = tgt_embeddings[tgt_word2id[word1]]
        else:
            print("wrong")
            continue
        if word2 in src_word2id:
            emb2 = src_embeddings[src_word2id[word2]]
        elif word2 in tgt_word2id:
            emb2 = tgt_embeddings[tgt_word2id[word2]]
        else:
            print("wrong")
            continue
        print(torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1))
    """
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


def load_xlm_embeddings(path, model_name="model"):
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

def load_map_embeddings(path, emb_size=-1):
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
            assert word not in word2id
            word2id[word] = len(word2id)
            emb = np.fromstring(emb, sep=' ', dtype=float)
            emb = torch.from_numpy(emb)
            embeddings.append(emb)
    embeddings = torch.stack(embeddings)
    id2word = {idx:word for word, idx in word2id.items()}
    return embeddings, id2word, word2id
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["xlm", "map"])
    parser.add_argument("--preprocess", default="ucu")
    parser.add_argument("--metric", default="nn")
    parser.add_argument("--dict", help="bilingual dictionary")
    if parser.parse_known_args()[0].method == "xlm":
        parser.add_argument("--reload", help="xlm model path")
        parser.add_argument("--model_name", type=str, choices=["model", "encoder", "decoder"], help='reload a xlm encoder or a xlm translation model')
        parser.add_argument("--use_language_suffix", action="store_true", default=False, help="use language suffix to detect language")
    elif parser.parse_known_args()[0].method == "map":
        parser.add_argument("--src_embs")
        parser.add_argument("--tgt_embs")
        parser.add_argument("--emb_size", type=int, default=-1)

    args = parser.parse_args()

    if args.method == "xlm":
        embeddings, dico = load_xlm_embeddings(args.reload, args.model_name)
        scores = eval_xlm_bli(embeddings, dico, args.dict, preprocess=args.preprocess, metric=args.metric, use_language_suffix=args.use_language_suffix)
    elif args.method == "map":
        src_embs, src_id2word, src_word2id = load_map_embeddings(args.src_embs, emb_size=args.emb_size)
        tgt_embs, tgt_id2word, tgt_word2id = load_map_embeddings(args.tgt_embs, emb_size=args.emb_size)
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
