import torch


class BaseSeq2Seq(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


    def get_loss_decoding_batch(self, encoded_info, decoder_inputs):
        """

        Args:
            encoded_info: class based on encoded_info
            decoder_inputs:  DecoderInputs which is fixed
        Returns:
            decoding_batch:  LossDecodingBatch

        """
        raise NotImplementedError

    def get_generate_batch(self, encoded_info, tgt_lang_id):
        """

        Args:
            encoded_info: class based on encoded_info
            tgt_lang_id: int
                language index

        Returns:
            generate_decode_batch: GenerateDecodeBatch
        """
        raise NotImplementedError

    def generate_and_run_loss(self, encoder_inputs, decoder_inputs, tgt_lang_id, decoding_params):
        """
        This is for evaluation
        Args:
            encoder_inputs:
            decoder_inputs:
        Returns:


        """
        encoded_info = self.encoder.encode(encoder_inputs)
        decoding_batch = self.get_loss_decoding_batch(encoded_info=encoded_info, decoder_inputs=decoder_inputs)
        scores, losses = self.run_decode_loss(decoding_batch, get_scores=True)

        generate_batch = self.get_generate_batch(encoded_info=encoded_info, tgt_lang_id=tgt_lang_id)
        generated, lengths = self.generate_decode(generate_decode_batch=generate_batch, decoding_params=decoding_params)
        return scores, losses, generated, lengths

    def run_seq2seq_loss(self, encoder_inputs, decoder_inputs, get_scores=False):
        encoded_info = self.encoder.encode(encoder_inputs)
        decoding_batch = self.get_loss_decoding_batch(encoded_info=encoded_info, decoder_inputs=decoder_inputs)
        scores, losses = self.run_decode_loss(decoding_batch, get_scores=get_scores)
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
    """
    Data to generate
    """
    def __init__(self, src_enc, src_len, tgt_lang_id, max_len, enc_mask):
        self.src_enc = src_enc
        self.src_len = src_len
        self.tgt_lang_id = tgt_lang_id
        self.max_len = max_len
        self.enc_mask = enc_mask


class DecoderInputs(object):
    """
    decoder inputs
    """

    def __init__(self, x2, len2, langs2, y, pred_mask, positions, lang_id):
        self.x2 = x2
        self.len2 = len2
        self.langs2 = langs2
        self.y = y
        self.pred_mask = pred_mask
        self.positions = positions
        self.langs2 = x2.clone().fill_(lang_id)
        self.lang_id = lang_id


class LossDecodingBatch(object):
    """
    informations to calculate decoding loss, encoder_info + decoder_inputs
    """

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

