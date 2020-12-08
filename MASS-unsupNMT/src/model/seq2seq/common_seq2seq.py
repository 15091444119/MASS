from . import  BaseSeq2Seq, DecoderInputs, LossDecodingBatch, GenerateDecodeBatch
from src.model.encoder.common_encoder import CommonEncodedInfo, CommonEncoder


class CommonSeq2Seq(BaseSeq2Seq):

    def __init__(self, encoder, decoder):
        assert isinstance(encoder, CommonEncoder)

        super().__init__(encoder=encoder, decoder=decoder)

    def forward(self, loss_name, encoder_inputs, decoder_inputs):
        if loss_name == "seq2seq_loss":
            return self.run_seq2seq_loss(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
        else:
            raise Exception("Unknown training method : {}".format(loss_name))

    def get_loss_decoding_batch(self, encoded_info: CommonEncodedInfo, decoder_inputs: DecoderInputs):
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

    def get_generate_batch(self, encoded_info: CommonEncodedInfo, tgt_lang_id):
        generate_batch = GenerateDecodeBatch(
            src_enc=encoded_info.encoded,
            src_len=encoded_info.enc_len,
            tgt_lang_id=tgt_lang_id,
            max_len=int(1.5 * encoded_info.enc_len.max().item() + 10),
            enc_mask=encoded_info.enc_mask
        )
        return generate_batch
