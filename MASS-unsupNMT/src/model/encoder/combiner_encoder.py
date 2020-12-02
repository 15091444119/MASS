from . import BaseEncoder, EncoderInputs
from src.combiner.combine_utils import CombineTool, ExplicitSplitCombineTool
import torch


class CombinerEncoder(BaseEncoder):

    def __init__(self, encoder, combiner, params, dico, splitter, loss_fn):
        super().__init__()
        self.encoder = encoder
        self.combiner = combiner
        self.params = params
        self.dico = dico
        self.splitter = splitter
        self.loss_fn = loss_fn

    def get_combine_tool(self, encoder_inputs):
        combine_tool = CombineTool(encoder_inputs.x1, length=encoder_inputs.len1, dico=self.dico, mask_index=self.params.mask_index)
        return combine_tool

    def explicit_encode_combiner_loss(self, encoder_inputs):

        explicit_batch = ExplicitSplitEncoderBatch(encoder_inputs, self.params, self.dico, self.splitter)
        combine_tool = ExplicitSplitCombineTool(
            splitted_batch=explicit_batch.x3,
            length_before_split=explicit_batch.len1,
            length_after_split=explicit_batch.len3,
            dico=self.dico,
            mappers=explicit_batch.mappers,
            mask_index=self.params.mask_index
        )

        # combine encode
        encoded = self.encoder(
            "fwd",
            x=explicit_batch.x3,
            lengths=explicit_batch.len3,
            langs=explicit_batch.langs3,
            causal=False
        ).transpose(0, 1)

        combined_rep = self.combiner.combine(
            encoded=encoded,
            lengths=combine_tool.final_length,
            combine_labels=combine_tool.combine_labels,
            lang_id=encoder_inputs.lang_id
        )

        final_encoded = combine_tool.gather(splitted_rep=encoded, combined_rep=combined_rep)

        # teacher
        if combine_tool.trained_combiner_words != 0:
            with torch.no_grad():
                original_rep = self.encoder(
                    "fwd",
                    x=explicit_batch.x1,
                    lengths=explicit_batch.len1,
                    langs=explicit_batch.langs1,
                    causal=False
                ).transpose(0, 1)

                dim = final_encoded.size(-1)

                trained_original_words_rep = original_rep.masked_select(
                    combine_tool.splitted_original_word_mask.unsqueeze(-1)).view(-1, dim)

            trained_combined_words_rep = combined_rep.index_select(
                dim=0,
                index=combine_tool.select_trained_rep_from_combined_rep
            ).view(-1, dim)

            combine_loss = self.loss_fn(trained_original_words_rep, trained_combined_words_rep)
        else:
            combine_loss = None

        return CombinerEncodedInfo(encoded=final_encoded, combine_tool=combine_tool), combine_loss, combine_tool.trained_combiner_words

    def encode(self, encoder_inputs: EncoderInputs):
        combine_tool = self.get_combine_tool(encoder_inputs)

        encoded = self.encoder(
            "fwd",
            x=encoder_inputs.x1,
            lengths=encoder_inputs.len1,
            langs=encoder_inputs.langs1,
            causal=False
        ).transpose(0, 1)

        combined_rep = self.combiner.combine(
            encoded=encoded,
            lengths=combine_tool.final_length,
            combine_labels=combine_tool.combine_labels,
            lang_id=encoder_inputs.lang_id
        )
        final_encoded = combine_tool.gather(splitted_rep=encoded, combined_rep=combined_rep)

        return CombinerEncodedInfo(encoded=final_encoded, combine_tool=combine_tool)


class CombinerEncodedInfo(object):

    def __init__(self, encoded, combine_tool):
        assert encoded.size(0) == combine_tool.final_length.size(0)
        assert encoded.size(1) == combine_tool.final_length.max().item()
        self.encoded = encoded
        self.enc_len = combine_tool.final_length
        self.enc_mask = combine_tool.mask_for_decoder
        self.original_len = combine_tool.original_length


class ExplicitSplitEncoderBatch(object):
    """
    Inputs for explicit_encode_combiner_loss
    """

    def __init__(self, encoder_inputs, params, dico, splitter):
        self.x1 = encoder_inputs.x1
        self.len1 = encoder_inputs.len1
        self.lang_id = encoder_inputs.lang_id
        self.langs1 = encoder_inputs.langs1

        # split whole words to train combiner
        x3, len3, mappers = splitter.re_encode_batch_sentences(batch=self.x1, lengths=self.len1, dico=dico, re_encode_rate=params.re_encode_rate)
        langs3 = x3.clone().fill_(self.lang_id)
        self.x3 = x3
        self.len3 = len3
        self.langs3 = langs3
        self.mappers = mappers
