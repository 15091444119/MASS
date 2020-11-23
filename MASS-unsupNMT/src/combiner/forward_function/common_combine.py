from .mass import set_model_mode
from ..combine_utils import CombineTool, ExplicitSplitCombineTool
from .explicit_split import ExplicitSplitEncoderBatch
import torch



class EncoderInputs(object):

    def __init__(self, x1, len1, lang_id, langs1):
        self.x1 = x1
        self.len1 = len1
        self.lang_id = lang_id
        self.langs1 = langs1

class CommonCombineModel(object):

    def __init__(self, encoder, decoder, combiner):
        self.encoder = encoder
        self.decoder = decoder
        self.combiner = combiner


class CommonCombineStat(object):

    def __init__(self, batch):
        self.processed_s = batch.len2.size(0)
        self.processed_w = (batch.len2 - 1).sum().item()


class CommonCombineLosses(object):

    def __init__(self, decoding_loss):
        self.decoding_loss = decoding_loss


class BaseEncoder(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class BaseCombinerEncoder(BaseEncoder):

    def __init__(self, encoder, combiner, params):
        super().__init__()
        self.encoder = encoder
        self.combiner = combiner
        self.params = params

    def convert_data(self, encoder_inputs):
        raise NotImplementedError

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def train_combiner(self):
        raise NotImplementedError


class BaseSeq2Seq(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def train(self, encoding_batch):
        pass

    def get_loss_decoding_batch(self, encoded_info, decoder_inputs):
        raise NotImplementedError

    def get_generate_batch(self, encoded_info, tgt_lang_id):
        raise NotImplementedError

    def run_seq2seq_loss(self, encoder_inputs, decoder_inputs):
        encoded_info = self.encoder.encode(encoder_inputs)
        decoding_batch = self.get_loss_decoding_batch(encoded_info=encoded_info, decoder_inputs=decoder_inputs)
        scores, losses = self.run_decode_loss(decoding_batch)
        return scores, losses

    def run_decode_loss(self, decoding_batch, get_scores=False):
        dec = self.decoder('fwd',
           x=decoding_batch.x,
           lengths=decoding_batch.lengths,
           langs=decoding_batch.langs,
           causal=True,
           src_enc=decoding_batch.src_enc,
           src_len=decoding_batch.src_len,
           positions=decoding_batch.positions,
           enc_mask=decoding_batch.src_mask
           )

        word_scores, loss = self.decoder('predict', tensor=dec, pred_mask=decoding_batch.pred_mask, y=decoding_batch.y, get_scores=get_scores)

        return word_scores, loss

    def generate(self, encoder_inputs, tgt_lang_id, decoding_params):
        encoded_info = self.encoder.encode(encoder_inputs)
        generate_batch = self.get_generate_batch(encoded_info=encoded_info, tgt_lang_id=tgt_lang_id)
        generated, lengths = self.generate_decode(generate_decode_batch=generate_batch, decoding_params=decoding_params)
        return generated, lengths

    def generate_decode(self, generate_decode_batch, decoding_params):
        if decoding_params.beam_size == 1:
            generated, lengths = self.decoder.generate(
                src_enc=generate_decode_batch.src_enc,
                src_len=generate_decode_batch.src_len,
                tgt_lang_id=generate_decode_batch.tgt_lang_id,
                max_len=generate_decode_batch.max_len,
                enc_mask=generate_decode_batch.enc_mask
            )
        else:
            generated, lengths = self.decoder.generate_beam(
                src_enc=generate_decode_batch.src_enc,
                src_len=generate_decode_batch.src_len,
                tgt_lang_id=generate_decode_batch.tgt_lang_id,
                beam_size=decoding_params.beam_size,
                length_penalty=decoding_params.length_penalty,
                early_stopping=decoding_params.early_stopping
            )
        return generated, lengths


class GenerateDecodeBatch(object):

    def __init__(self, src_enc, src_len, tgt_lang_id, max_len, enc_mask):
        self.src_enc = src_enc
        self.src_len = src_len
        self.tgt_lang_id = tgt_lang_id
        self.max_len = max_len
        self.enc_mask = enc_mask


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


class DecodeInputBatch(object):

    def __init__(self, x2, len2, y, pred_mask, positions, lang_id):
        self.x2 = x2
        self.len2 = len2
        self.y = y
        self.pred_mask = pred_mask
        self.positions = positions
        self.langs2 = x2.clone().fill_(lang_id)
        self.lang_id = lang_id


class LossDecodingBatch(object):

    def __init__(self, x, lengths, langs, src_enc, src_len, src_mask, positions, pred_mask, y, lang_id):
        self.x = x
        self.lengths = lengths
        self.langs = langs
        self.src_enc = src_enc
        self.src_len = src_len
        self.src_mask = src_mask
        self.positions = positions
        self.pred_mask = pred_mask
        self.y = y
        self.lang_id = lang_id


class CommonSeq2Seq(BaseSeq2Seq):

    def __init__(self, encoder, decoder):
        super().__init__(encoder=encoder, decoder=decoder)

    def get_loss_decoding_batch(self, encoded_info: CommonEncodedInfo, decoder_inputs: DecodeInputBatch):
        loss_decoding_batch = LossDecodingBatch(
            x=decoder_inputs.x2,
            lengths=decoder_inputs.len2,
            langs=decoder_inputs.langs2,
            src_enc=encoded_info.encoded,
            src_len=encoded_info.enc_len,
            src_mask=None,
            positions=decoder_inputs.positions,
            pred_mask=decoder_inputs.pred_mask,
            y=decoder_inputs.y,
            lang_id=decoder_inputs.lang_id
        )
        return loss_decoding_batch

    def get_generate_batch(self, encoded_info: CommonEncodedInfo, tgt_lang_id):
        generate_batch = GenerateDecodeBatch(
            src_enc=encoded_info.encoded,
            src_len=encoded_info.enc_len,
            tgt_lang_id=tgt_lang_id,
            max_len=int(1.5 * encoded_info.enc_len.max().item() + 10),
            enc_mask=None
        )
        return generate_batch


class CombinerEncoder(BaseEncoder):

    def __init__(self, encoder, combiner, params, dico, splitter, loss_fn):
        super().__init__()
        self.encoder = encoder
        self.combiner = combiner
        self.params = params
        self.dico = dico
        self.splitter = splitter
        self.loss_fn = loss_fn

    def get_combine_tool(self, encode_inputs):
        combine_tool = CombineTool(encode_inputs.x1, length=encode_inputs.len1, dico=self.dico, mask_index=self.params.mask_index)
        return combine_tool

    def explicit_encode_combiner_loss(self, encode_inputs):

        explicit_batch = ExplicitSplitEncoderBatch(encode_inputs, self.params, self.dico, self.splitter)
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
            lang_id=encode_inputs.lang_id
        )

        final_encoded = combine_tool.gather(splitted_rep=encoded, combined_rep=combined_rep)

        # teacher
        if combine_tool.trained_combiner_words != 0:
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

            trained_combined_words_rep = combined_rep.index_select(dim=0,
                                                                   index=combine_tool.select_trained_rep_from_combined_rep).view(-1, dim)

            combine_loss = self.loss_fn(trained_original_words_rep, trained_combined_words_rep)
        else:
            combine_loss = None

        return CombinerEncodedInfo(encoded=final_encoded, combine_tool=combine_tool), combine_loss

    def encode(self, encode_inputs: EncoderInputs):
        combine_tool = self.get_combine_tool(encode_inputs)

        encoded = self.encoder(
            "fwd",
            x=encode_inputs.x1,
            lengths=encode_inputs.len1,
            langs=encode_inputs.langs1,
            causal=False
        ).transpose(0, 1)

        combined_rep = self.combiner.combine(
            encoded=encoded,
            lengths=combine_tool.final_length,
            combine_labels=combine_tool.combine_labels,
            lang_id=encode_inputs.lang_id
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


class CombineSeq2Seq(BaseSeq2Seq):

    def __init__(self, encoder:CombinerEncoder, decoder):
        super().__init__(encoder=encoder, decoder=decoder)

    def explicit_loss(self, encode_inputs, decoder_inputs, get_scores=False):
        encoded_info, combiner_loss = self.encoder.explicit_encode_combiner_loss(encode_inputs=encode_inputs)
        decoding_batch = self.get_loss_decoding_batch(encoded_info=encoded_info, decoder_inputs=decoder_inputs)
        scores, losses = self.run_decode_loss(decoding_batch=decoding_batch, get_scores=get_scores)
        return scores, losses

    def get_loss_decoding_batch(self, encoded_info: CombinerEncodedInfo, decoder_inputs):
        loss_decoding_batch = LossDecodingBatch(
            x=decoder_inputs.x2,
            lengths=decoder_inputs.len2,
            langs=decoder_inputs.langs2,
            src_enc=encoded_info.encoded,
            src_len=encoded_info.enc_len,
            src_mask=encoded_info.enc_mask,
            positions=decoder_inputs.positions,
            pred_mask=decoder_inputs.pred_mask,
            y=decoder_inputs.y,
            lang_id=decoder_inputs.lang_id
        )
        return loss_decoding_batch

    def get_generate_batch(self, encoded_info: CombinerEncodedInfo, tgt_lang_id):
        generate_batch = GenerateDecodeBatch(
            src_enc=encoded_info.encoded,
            src_len=encoded_info.enc_len,
            tgt_lang_id=tgt_lang_id,
            max_len=int(1.5 * encoded_info.original_len.max().item() + 10),
            enc_mask=None
        )
        return generate_batch

