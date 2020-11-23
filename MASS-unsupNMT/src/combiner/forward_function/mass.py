
class DecodeInputBatch(object):

    def __init__(self, x2, len2, y, pred_mask, positions, lang_id):
        self.x2 = x2
        self.len2 = len2
        self.y = y
        self.pred_mask = pred_mask
        self.positions = positions
        self.langs2 = x2.clone().fill_(lang_id)
        self.lang_id = lang_id


class EncoderInputs(object):

    def __init__(self, x1, len1, lang_id, langs1):
        self.x1 = x1
        self.len1 = len1
        self.lang_id = lang_id
        self.langs1 = langs1


class DecodingBatch(object):

    def __init__(self, x, length, langs, src_enc, src_len, src_mask, positions, pred_mask, y, lang_id):
        self.x = x
        self.length = length
        self.langs = langs
        self.src_enc = src_enc
        self.src_len = src_len
        self.src_mask = src_mask
        self.positions = positions
        self.pred_mask = pred_mask
        self.y = y
        self.lang_id = lang_id


def set_model_mode(mode, models):
    if mode == "train":
        for model in models:
            model.train()
    elif mode == "eval":
        for model in models:
            model.eval()
    else:
        raise ValueError

