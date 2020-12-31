from . import BaseEncoder, EncoderInputs, EncodedInfo
from src.model.context_combiner.combine_utils import CombineTool, ExplicitSplitCombineTool
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
            lengths=explicit_batch.len3,
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

        return EncodedInfo(encoded=final_encoded, enc_mask=combine_tool.mask_for_decoder, enc_len=combine_tool.final_length), combine_loss, combine_tool.trained_combiner_words

    def encode(self, encoder_inputs: EncoderInputs):
        combine_tool = CombineTool(encoder_inputs.x1, length=encoder_inputs.len1, dico=self.dico, mask_index=self.params.mask_index)

        encoded = self.encoder(
            "fwd",
            x=encoder_inputs.x1,
            lengths=encoder_inputs.len1,
            langs=encoder_inputs.langs1,
            causal=False
        ).transpose(0, 1)

        combined_rep = self.combiner.combine(
            encoded=encoded,
            lengths=encoder_inputs.len1,
            combine_labels=combine_tool.combine_labels,
            lang_id=encoder_inputs.lang_id
        )
        final_encoded = combine_tool.gather(splitted_rep=encoded, combined_rep=combined_rep)

        return EncodedInfo(encoded=final_encoded, enc_mask=combine_tool.mask_for_decoder, enc_len=combine_tool.final_length)


class MultiCombinerEncoder(BaseEncoder):
    def __init__(self, encoder, lang_id2combiner, dico, mask_index):
        super().__init__()
        self.encoder = encoder
        self.lang_id2combiner = torch.nn.ModuleDict(lang_id2combiner)
        self.dico = dico
        self.mask_index = mask_index

    def encode(self, encoder_inputs: EncoderInputs):
        encoder = self.encoder
        combiner = self.lang_id2combiner[encoder_inputs.lang_id]

        combine_tool = CombineTool(encoder_inputs.x1, length=encoder_inputs.len1, dico=self.dico, mask_index=self.mask_index)

        encoded = encoder(
            "fwd",
            x=encoder_inputs.x1,
            lengths=encoder_inputs.len1,
            langs=encoder_inputs.langs1,
            causal=False
        ).transpose(0, 1)

        combined_rep = combiner.combine(
            encoded=encoded,
            lengths=encoder_inputs.len1,
            combine_labels=combine_tool.combine_labels,
        )
        final_encoded = combine_tool.gather(splitted_rep=encoded, combined_rep=combined_rep)

        return EncodedInfo(encoded=final_encoded, enc_mask=combine_tool.mask_for_decoder, enc_len=combine_tool.final_length)


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
