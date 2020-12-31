from . import BaseSeq2Seq, DecoderInputs, LossDecodingBatch, GenerateDecodeBatch


class CommonSeq2Seq(BaseSeq2Seq):

    def __init__(self, encoder, decoder):

        super().__init__(encoder=encoder, decoder=decoder)

    def forward(self, loss_name, encoder_inputs, decoder_inputs):
        if loss_name == "seq2seq_loss":
            return self.run_seq2seq_loss(encoder_inputs=encoder_inputs, decoder_inputs=decoder_inputs)
        else:
            raise Exception("Unknown training method : {}".format(loss_name))
