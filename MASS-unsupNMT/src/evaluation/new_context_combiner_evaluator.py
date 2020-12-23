import torch
from src.trainer.new_context_combiner_trainer import combiner_step

from logging import getLogger

logger = getLogger()


class NewContextCombinerEvaluator(object):

    def __init__(self, data, params, encoder, combiner, pred_layer, loss_fn, lang_id):
        self.data = data
        self.params = params
        self.encoder = encoder
        self.combiner = combiner
        self.loss_fn = loss_fn
        self.lang_id = lang_id
        self.pred_layer = pred_layer

    def run_all_evals(self, epoch):
        scores = {}
        scores["epoch"] = epoch
        self.eval_loss(scores)
        return scores

    def eval_loss(self, scores, save_loss=None):
        """

        Args:
            scores:
            save_losses: save loss to a file

        Returns:

        """
        self.combiner.eval()
        self.encoder.eval()
        self.pred_layer.eval()


        with torch.no_grad():
            for part in ["dev", "test"]:
                loss_sum = 0
                n_words = 0
                n_right_words = 0
                id2losses = {}
                id2results = {}

                for batch_id, batch in enumerate(self.data[part]):

                    loss, pred_results, trained_sentences, trained_words = combiner_step(
                        encoder=self.encoder,
                        combiner=self.combiner,
                        lang_id=self.lang_id,
                        batch=batch,
                        loss_fn=self.loss_fn,
                        pred_layer=self.pred_layer
                    )

                    n_words += trained_words
                    n_right_words += (pred_results).long().sum().item()
                    loss_sum += loss.mean().item() * trained_words

                    # update loss for each word
                    update_id2losses(id2losses=id2losses, loss=loss, batch=batch)
                    update_id2results(id2results=id2results, results=pred_results, batch=batch)

                    if batch_id % 100 == 0:
                        logger.info("{} words".format(n_words))

                id2average_loss = {idx: sum(losses) / len(losses) for idx, losses in id2losses.items()}
                if save_loss is not None:
                    save_id2average_loss(id2average_loss, self.data["dico"], save_loss)

                id2acc = {idx: sum(results) / len(results) for idx, results in id2results.items()}
                scores["{}-word-average-acc".format(part)] = sum(id2acc.values()) / len(id2acc.values())

                scores["{}-word-average-loss".format(part)] = sum(id2average_loss.values()) / len(id2average_loss.values())

                scores["{}-loss".format(part)] = loss_sum / n_words

                scores["{}-acc".format(part)] = n_right_words / n_words


def update_id2results(id2results, results, batch):
    trained_words = batch["original_batch"].transpose(0, 1).masked_select(batch["trained_word_mask"])
    for idx, dict_id in enumerate(trained_words):
        dict_id = dict_id.item()
        result = (results[idx]).long().sum().item()  # 01 resutls
        if dict_id in id2results:
            id2results[dict_id].append(result)
        else:
            id2results[dict_id] = [result]


def update_id2losses(id2losses, loss, batch):
    trained_words = batch["original_batch"].transpose(0, 1).masked_select(batch["trained_word_mask"])
    for idx, dict_id in enumerate(trained_words):
        dict_id = dict_id.item()
        if dict_id in id2losses:
            id2losses[dict_id].append(loss[idx].item())
        else:
            id2losses[dict_id] = [loss[idx].item()]


def save_id2average_loss(id2average_loss, dico, path):
    id2average_loss = list(id2average_loss.items())
    sorted_id2average_loss = sorted(id2average_loss, key=lambda x: x[1], reverse=True)
    with open(path, 'w') as f:
        for idx, loss in sorted_id2average_loss:
            print(dico.id2word[idx], loss, file=f)
