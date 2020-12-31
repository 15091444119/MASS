import torch


class BaseEncoder(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class EncoderInputs(object):

    def __init__(self, x1, len1, lang_id, langs1):
        self.x1 = x1
        self.len1 = len1
        self.lang_id = lang_id
        self.langs1 = langs1


class EncodedInfo(object):

    def __init__(self, encoded, enc_len, enc_mask):
        assert encoded.size(0) == enc_len.size(0)
        assert encoded.size(1) == max(enc_len)
        self.encoded = encoded
        self.enc_len = enc_len
        self.enc_mask = enc_mask
