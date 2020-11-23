"""
cheat means we use original whole word representation as combined results
"""


from .mass import set_model_mode
from .explicit_split import ExplicitSplitCombineTool, ExplicitSplitEncoderBatch
from ..combine_utils import CheatCombineTool
from .common_combine import BaseCombinerEncoder, DecodeInputBatch, BaseEncoder, BaseSeq2Seq, EncoderInputs, CommonEncoder, LossDecodingBatch, GenerateDecodeBatch


class CheatEncoder(BaseEncoder):

    def __init__(self, encoder, params, dico, splitter):
        super().__init__()
        self.encoder = encoder
        self.params = params
        self.dico = dico
        self.splitter = splitter

    def convert_data(self, encode_inputs: EncoderInputs):
        batch = ExplicitSplitEncoderBatch(encode_inputs, params=self.params, dico=self.dico, splitter=self.splitter)
        combine_tool = CheatCombineTool(
            splitted_batch=batch.x3,
            length_before_split=batch.len1,
            length_after_split=batch.len3,
            dico=self.dico,
            mappers=batch.mappers,
            mask_index=self.params.mask_index
        )
        return batch, combine_tool

    def encode(self, encode_inputs: EncoderInputs):
        explicit_split_batch, combine_tool = self.convert_data(encode_inputs)
        orignal_encoded = self.encoder(
            "fwd",
            x=explicit_split_batch.x1,
            lengths=explicit_split_batch.len1,
            langs=explicit_split_batch.langs1,
            causal=False
        ).transpose(0, 1)  # [bs, len, dim]

        splitted_encoded = self.encoder(
            "fwd",
            x=explicit_split_batch.x3,
            lengths=explicit_split_batch.len3,
            langs=explicit_split_batch.langs3,
            causal=False
        ).transpose(0, 1)  # [bs, len, dim]

        # cheat!
        dim = orignal_encoded.size(-1)
        cheated_combine_rep = orignal_encoded.masked_select(
            combine_tool.splitted_original_word_mask.unsqueeze(-1)).view(-1, dim)

        encoded = combine_tool.gather(splitted_rep=splitted_encoded, combined_rep=cheated_combine_rep)

        return CombinerEncodedInfo(encoded=encoded, combine_tool=combine_tool)


