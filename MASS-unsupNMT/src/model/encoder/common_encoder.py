from . import BaseEncoder, EncoderInputs, EncodedInfo


class CommonEncoder(BaseEncoder):

    def __init__(self, model, mask_index):
        super().__init__()
        self.model = model
        self.mask_index = mask_index

    def encode(self, encoder_inputs: EncoderInputs):
        encoded = self.model(
            "fwd",
            x=encoder_inputs.x1,
            lengths=encoder_inputs.len1,
            langs=encoder_inputs.langs1,
            causal=False
        ).transpose(0, 1)  # [bs, len, dim]

        enc_mask = encoder_inputs.x1.ne(self.mask_index).transpose(0, 1)

        return EncodedInfo(encoded=encoded, enc_len=encoder_inputs.len1, enc_mask=enc_mask)

