from . import BaseEncoder


class CommonEncoder(BaseEncoder):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode(self, encoder_inputs):
        encoded = self.model(
            "fwd",
            x=encoder_inputs.x1,
            lengths=encoder_inputs.len1,
            langs=encoder_inputs.langs1,
            causal=False
        ).transpose(0, 1)  # [bs, len, dim]
        return CommonEncodedInfo(encoded=encoded, enc_len=encoder_inputs.len1)


class CommonEncodedInfo(object):

    def __init__(self, encoded, enc_len):
        assert encoded.size(0) == enc_len.size(0)
        assert encoded.size(1) == max(enc_len)
        self.encoded = encoded
        self.enc_len = enc_len