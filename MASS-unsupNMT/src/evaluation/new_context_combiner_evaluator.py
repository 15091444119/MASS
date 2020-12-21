import torch
from src.trainer.new_context_combiner_trainer import combiner_step


class NewContextCombinerEvaluator(object):

    def __init__(self, data, params, encoder, combiner, loss_fn, lang_id):
        self.data = data
        self.params = params
        self.encoder = encoder
        self.combiner = combiner
        self.loss_fn = loss_fn
        self.lang_id = lang_id

    def run_all_evals(self, epoch):
        scores = {}
        scores["epoch"] = epoch
        self.eval_loss(scores)
        return scores

    def eval_loss(self, scores):
        self.combiner.eval()
        self.encoder.eval()
        with torch.no_grad():
            for part in ["dev", "test"]:
                loss_sum = 0
                n_words = 0

                for batch in self.data[part]:
                    loss, trained_sentences, trained_words = combiner_step(
                        encoder=self.encoder,
                        combiner=self.combiner,
                        lang_id=self.lang_id,
                        batch=batch,
                        loss_fn=self.loss_fn
                    )

                    n_words += trained_words
                    loss_sum += loss * trained_words

                scores["combiner-loss".format(part)] = loss_sum / n_words
