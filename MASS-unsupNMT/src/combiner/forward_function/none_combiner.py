from .common_combine import BaseCombinerEncoder
from .mass import EncoderInputs

class NoneCombinerTool(object):

    def __init__(self, batch: EncoderInputs):
        self.original_length = batch.len1
        self.final_length = batch.len1
        self.enc_mask = None


class NoneCombinerEncoder(BaseCombinerEncoder):

    def __init__(self, encoder, combiner, params):
        super().__init__(encoder, combiner, params)

    def convert_data(self, batch: EncoderInputs):
        return batch, NoneCombinerTool(batch)

    def encode(self, batch: EncoderInputs, combine_tool:NoneCombinerTool):
        return self.encoder(
            "fwd",
            x=batch.x1,
            lengths=batch.len1,
            langs=batch.langs1,
            causal=False
        ).transpose(0, 1)  # [len, bs, dim]
