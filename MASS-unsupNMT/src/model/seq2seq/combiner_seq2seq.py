from . import LossDecodingBatch, GenerateDecodeBatch, BaseSeq2Seq
from src.model.encoder.combiner_encoder import CombinerEncodedInfo, CombinerEncoder


class CombinerSeq2Seq(BaseSeq2Seq):

    def __init__(self, encoder:CombinerEncoder, decoder):
        """
        Args:
            encoder:  CombinerEncoder
            decoder:  mass decoder
        """
        super().__init__(encoder=encoder, decoder=decoder)

    def forward(self, loss_name, encoder_inputs, decoder_inputs):
        if loss_name == "explicit_loss":
             return self.explicit_loss(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
        elif loss_name == "seq2seq_loss":
            return self.run_seq2seq_loss(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
        else:
            raise Exception("Unknown training method : {}".format(loss_name))

    def explicit_loss(self, encoder_inputs, decoder_inputs, get_scores=False):
        """

        Args:
            encoder_inputs:  EncodeInputs
            decoder_inputs: DecoderInputs
            get_scores: Bool
                if set to True, return scores, else scores is None
        Returns:
            scores, losses
        """
        encoded_info, combiner_loss, trained_combiner_words = self.encoder.explicit_encode_combiner_loss(encoder_inputs=encoder_inputs)
        decoding_batch = self.get_loss_decoding_batch(encoded_info=encoded_info, decoder_inputs=decoder_inputs)
        scores, losses = self.run_decode_loss(decoding_batch=decoding_batch, get_scores=get_scores)
        return scores, losses, combiner_loss, trained_combiner_words

    def get_loss_decoding_batch(self, encoded_info: CombinerEncodedInfo, decoder_inputs):
        """
        Args:
            encoded_info:  CombinerEncodedInfo
            decoder_inputs: DecoderInputs
        Returns:
            loss_decoding_batch
        """
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
