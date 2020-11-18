
class MassBatch(object):

    def __init__(self, x1, len1, x2, len2, y, pred_mask, positions, lang):
        self.x1 = x1
        self.len1 = len1
        self.x2 = x2
        self.len2 = len2
        self.y = y
        self.pred_mask = pred_mask
        self.positions = positions
        self.lang = lang


class DecodingBatch(object):

    def __init__(self, x, length, langs, src_enc, src_len, src_mask, positions, pred_mask, y):
        self.x = x
        self.length = length
        self.langs = langs
        self.src_enc = src_enc
        self.src_len = src_len
        self.src_mask = src_mask
        self.positions = positions
        self.pred_mask = pred_mask
        self.y = y

    def decode(self, decoder, get_scores):
        dec2 = decoder('fwd',
                       x=self.x,
                       lengths=self.length,
                       langs=self.langs,
                       causal=True,
                       src_enc=self.src_enc,
                       src_len=self.src_len,
                       positions=self.positions,
                       enc_mask=self.src_mask
                       )

        word_scores, loss = decoder('predict', tensor=dec2, pred_mask=self.pred_mask, y=self.y, get_scores=get_scores)

        return word_scores, loss


def set_model_mode(mode, models):
    if mode == "train":
        for model in models:
            model.train()
    elif mode == "eval":
        for model in models:
            model.eval()
    else:
        raise ValueError

