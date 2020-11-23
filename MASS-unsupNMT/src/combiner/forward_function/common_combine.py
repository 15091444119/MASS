from .mass import set_model_mode, DecodingBatch
from ..combine_utils import CombineTool
import torch



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


class CommonCombineBatch(object):

    def __init__(self, mass_batch, params):
        self.x1 = mass_batch.x1
        self.len1 = mass_batch.len1
        self.x2 = mass_batch.x2
        self.len2 = mass_batch.len2
        self.y = mass_batch.y
        self.pred_mask = mass_batch.pred_mask
        self.positions = mass_batch.positions
        self.lang_id = mass_batch.lang_id
        self.langs1 = mass_batch.langs1
        self.langs2 = mass_batch.langs2


class BaseCombinerEncoder(torch.nn.Module):

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
        raise  NotImplementedError


class GenerateDecodeBatch(object):

    def __init__(self, src_enc, src_len, tgt_lang_id, max_len, enc_mask):
        self.src_enc = src_enc
        self.src_len = src_len
        self.tgt_lang_id = tgt_lang_id
        self.max_len = max_len
        self.enc_mask = enc_mask


class CombinerEncoderDecoder(torch.nn.Module):

    def __init__(self, combiner_encoder: BaseCombinerEncoder, decoder):
        super().__init__()
        self.combiner_encoder = combiner_encoder
        self.decoder = decoder

    def train(self, encoding_batch):
        pass

    def mass_loss(self, encoder_inputs, decoder_inputs):
        batch, combine_tools = self.combiner_encoder.convert_data(encoder_inputs)
        encoded = self.combiner_encoder.encode(batch, combine_tools)
        decoding_batch = self.get_loss_decoding_batch(encoded, decoder_inputs, combine_tools)
        scores, losses = self.decode(decoding_batch)

        return scores, losses

    def loss_decode(self, decoding_batch, get_scores=False):
        dec = self.decoder('fwd',
           x=decoding_batch.x,
           lengths=decoding_batch.length,
           langs=decoding_batch.langs,
           causal=True,
           src_enc=decoding_batch.src_enc,
           src_len=decoding_batch.src_len,
           positions=decoding_batch.positions,
           enc_mask=decoding_batch.src_mask
           )

        word_scores, loss = self.decoder('predict', tensor=dec, pred_mask=decoding_batch.pred_mask, y=decoding_batch.y, get_scores=get_scores)

        return word_scores, loss

    def get_generate_decode_batch(self, encoded, tgt_lang_id, combine_tools):
        max_len =int(1.5 * combine_tools.original_length.max().item() + 10)
        generate_decode_batch = GenerateDecodeBatch(
            src_enc=encoded,
            src_len=combine_tools.final_length,
            tgt_lang_id=tgt_lang_id,
            max_len=max_len,
            enc_mask=combine_tools.enc_mask
        )
        return generate_decode_batch

    def generate(self, encoder_inputs, tgt_lang_id, decoding_params):
        batch, combine_tools = self.combiner_encoder.convert_data(encoder_inputs)
        encoded = self.combiner_encoder.encode(batch, combine_tools)
        generate_decode_batch = self.get_generate_decode_batch(encoded, tgt_lang_id, combine_tools)
        generated, lengths = self.generate_decode(generate_decode_batch=generate_decode_batch, decoding_params=decoding_params)
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



def combiner_mass(models, mass_batch, params, dico, mode):
    # prepare
    set_model_mode(mode=mode, models=[models.encoder, models.combiner, models.decoder])

    batch = CommonCombineBatch(mass_batch=mass_batch, params=params)

    combine_tool = CombineTool(
        batch=batch.x1,
        length=batch.len1,
        dico=dico,
        mask_index=params.mask_index
    )

    # combine and maybe train combiner
    final_encoded = encode(common_combine_batch=batch,
                           combiner=models.combiner,
                           encoder=models.encoder,
                           combine_tool=combine_tool
    )

    # decoding
    decoding_batch = DecodingBatch(
        x=batch.x2,
        length=batch.len2,
        langs=batch.langs2,
        src_enc=final_encoded,
        src_len=combine_tool.final_length,
        src_mask=combine_tool.mask_for_decoder,
        positions=batch.positions,
        pred_mask=batch.pred_mask,
        y=batch.y,
        lang_id=batch.lang_id
    )

    scores, decoding_loss = decoding_batch.decode(decoder=models.decoder, get_scores=(mode == "eval"))

    losses = CommonCombineLosses(decoding_loss=decoding_loss)

    statistics = CommonCombineStat(batch=batch)

    return scores, losses, statistics


def encode(common_combine_batch, combiner, encoder, combine_tool):
    encoded = encoder(
        "fwd",
        x=common_combine_batch.x1,
        lengths=common_combine_batch.len1,
        langs=common_combine_batch.langs1,
        causal=False
    ).transpose(0, 1)

    combined_rep = combiner.combine(
        encoded=encoded,
        lengths=combine_tool.final_length,
        combine_labels=combine_tool.combine_labels,
        lang_id=common_combine_batch.lang_id
    )

    final_encoded = combine_tool.gather(splitted_rep=encoded, combined_rep=combined_rep)

    return final_encoded

