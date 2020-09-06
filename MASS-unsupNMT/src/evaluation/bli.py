""" bilingual lexicion functions """
import torch
import pdb
import argparse
import numpy as np
import sys


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
            src, tgt = line.rstrip().split(' ')
            if src in src_word2id and tgt in tgt_word2id:
                src_id = src_word2id[src]
                if src_id in dic:
                    if tgt_word2id[tgt] in dic[src_id]:
                        dropped_pair_count += 1
                    else:
                        dic[src_id].append(tgt_word2id[tgt])
                else:
                    dic[src_id] = [tgt_word2id[tgt]]
            else:
                dropped_pair_count += 1
    print("Dropped {} pairs, resulting in dictionary of {} source words".format(dropped_pair_count, len(dic)), file=sys.stderr)
    return dic


class BLI(object):
    """ object for bilingual dictionary induction """
    def __init__(self, preprocess_method, batch_size=500, metric="nn", csls_topk=10):
        self._preprocess_method = preprocess_method
        self._batch_size = batch_size
        self._metric = metric
        self._csls_topk = csls_topk

    def eval(self, src_embeddings, tgt_embeddings, src_id2word, src_word2id, tgt_id2word, tgt_word2id, dic):
        """Evaluate bilingual dictionary induction
        Params:
            dic(dict): dict from src id to tgt id list
        """
        if torch.cuda.is_available():
            src_embeddings = src_embeddings.cuda()
            tgt_embeddings = tgt_embeddings.cuda()

        if self._preprocess_method != "":
            src_embeddings = preprocess_embedding(src_embeddings, self._preprocess_method)
            tgt_embeddings = preprocess_embedding(tgt_embeddings, self._preprocess_method)

        print(
            "Src embedding size:{} Tgt embedding size:{} Src words in dictionary:{} Tgt words in dictionary:{}".format(
                src_embeddings.size(0),
                tgt_embeddings.size(0),
                len(list(dic.keys())),
                sum([len(value) for key, value in dic.items()])
            ),
            file=sys.stderr
        )

        translation = retrieval(src_embeddings, tgt_embeddings, list(dic.keys()), csls_topk=self._csls_topk,
                                batch_size=self._batch_size, metric=self._metric)

        top1_acc = calculate_word_translation_accuracy(translation, dic, topk=1)
        top5_acc = calculate_word_translation_accuracy(translation, dic, topk=5)
        top10_acc = calculate_word_translation_accuracy(translation, dic, topk=10)

        scores = {"top1_acc": top1_acc, "top5_acc": top5_acc, "top10_acc": top10_acc}

        return scores

    def translate_words(
        self,
        src_embeddings,
        tgt_embeddings,
        src_id2word,
        src_word2id,
        tgt_id2word,
        tgt_word2id,
        src_words_to_translate,
        print_translation=True):
        """ translate words in src_words to translate, words not in src_word2id won't be translated"""

        if torch.cuda.is_available():
            src_embeddings = src_embeddings.cuda()
            tgt_embeddings = tgt_embeddings.cuda()

        if self._preprocess_method != "":
            src_embeddings = preprocess_embedding(src_embeddings, self._preprocess_method)
            tgt_embeddings = preprocess_embedding(tgt_embeddings, self._preprocess_method)

        src_ids_to_translate = []
        for src_word in src_words_to_translate:
            if src_word in src_word2id:
                src_ids_to_translate.append(src_word2id[src_word])
            else:
                print("Unseen source word: {}".format(src_word), file=sys.stderr)

        translation = retrieval(src_embeddings, tgt_embeddings, src_ids_to_translate, csls_topk=self._csls_topk, batch_size=self._batch_size, metric=self._metric)

        if print_translation:
            for src_id, tgt_list in translation.items():
                for idx, tgt_id in enumerate(tgt_list):
                    print("{}-> {} {}".format(src_id2word[src_id], idx + 1, tgt_id2word[tgt_id]))

        return translation
