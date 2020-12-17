from .bli import BLI
import torch
import pdb

class EmbCombinerEvaluator(object):

    def __init__(self, data, params, combiner, embeddings, loss_fn):
        self.data = data
        self.params = params
        self.combiner = combiner
        self.embeddings = embeddings
        self.loss_fn = loss_fn
        self.bli = BLI(preprocess_method="u", metric="nn")

    def run_all_evals(self, epoch):
        scores = {}
        scores["epoch"] = epoch
        self.eval_loss(scores)
        self.eval_neighbor_search(scores)
        return scores

    def eval_loss(self, scores):
        self.combiner.eval()
        self.embeddings.eval()
        with torch.no_grad():
            for part in ["dev", "test"]:
                loss_sum = 0
                n_words = 0

                for batch in self.data[part]:
                    batch_splitted_word_ids, splitted_words_lengths, batch_whole_word_id = batch

                    batch_splitted_word_embeddings = self.embeddings(batch_splitted_word_ids).detach()

                    combiner_rep = self.combiner(batch_splitted_word_embeddings, splitted_words_lengths)

                    original_rep = self.embeddings(batch_whole_word_id).detach()  # don't update embeddings

                    loss = self.loss_fn(combiner_rep, original_rep)

                    loss_sum += loss.item() * batch_whole_word_id.size(0)
                    n_words += batch_whole_word_id.size(0)

                scores["{}-combiner-loss".format(part)] = loss_sum / n_words

    def eval_neighbor_search(self, scores):
        self.combiner.eval()
        self.embeddings.eval()
        with torch.no_grad():

            for part in ["dev", "test"]:
                src_embeddings = []
                src_ids = []

                for batch in self.data[part]:
                    batch_splitted_word_ids, splitted_words_lengths, batch_whole_word_id = batch

                    batch_splitted_word_embeddings = self.embeddings(batch_splitted_word_ids).detach()
                    combiner_rep = self.combiner(batch_splitted_word_embeddings, splitted_words_lengths)
                    src_embeddings.append(combiner_rep)
                    src_ids.extend(list(batch_whole_word_id.cpu().numpy()))

                src_embeddings = torch.cat(src_embeddings, dim=0)
                src_id2word = build_new_id2word(src_ids, self.data["dico"])
                src_word2id = {word: id for id, word in src_id2word.items()}
                dic = {i: [src_ids[i]] for i in range(len(src_id2word))}

                bli_scores = self.bli.eval(
                    src_embeddings=src_embeddings,
                    src_id2word=src_id2word,
                    src_word2id=src_word2id,
                    tgt_embeddings=self.embeddings.weight.detach(),
                    tgt_id2word=self.data["dico"].id2word,
                    tgt_word2id=self.data["dico"].word2id,
                    dic=dic
                )

                for key, value in bli_scores.items():
                    scores["NeighborSearchAcc-{}-{}".format(part, key)] = value


def build_new_id2word(src_ids, dico):
    """

    Args:
        src_ids: ids  in dico
        dico:

    Returns:
        id2word: position in src_ids to it's word
    """
    id2word = {}

    for position, id in enumerate(src_ids):
        id2word[position] = dico.id2word[id]

    return id2word

