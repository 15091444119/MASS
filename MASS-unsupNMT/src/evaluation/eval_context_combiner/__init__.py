import argparse
import torch
import numpy as np
from src.model.encoder import EncoderInputs
from src.utils import to_cuda
from src.model import load_combiner_model
from src.combiner.combiner import AverageCombiner
from .dataset import AlignmentDataset, AlignmentTypes


def hack(mass_checkpoint, combiner_seq2seq):
    """
    1.  use average combiner instead of the original combiner
    2. use the params in the mass checkpoint
    Args:
        mass_checkpoint:
        combiner_seq2seq:

    Returns:
    """
    print("HACK to Average combiner")
    combiner_seq2seq.encoder.combiner = AverageCombiner()
    if mass_checkpoint != "":
        reloaded = torch.load(mass_checkpoint)
        enc_reload = reloaded['encoder']
        if all([k.startswith('module.') for k in enc_reload.keys()]):
            enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}
        dec_reload = reloaded['decoder']
        if all([k.startswith('module.') for k in dec_reload.keys()]):
            dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}
        combiner_seq2seq.encoder.encoder.load_state_dict(enc_reload)
        combiner_seq2seq.decoder.load_state_dict(dec_reload)
    return combiner_seq2seq


def get_encoder_inputs(x, len, lang_id):

    langs = x.clone().fill_(lang_id)
    x, len, langs = to_cuda(x, len, langs)
    encoder_inputs = EncoderInputs(x1=x, len1=len, lang_id=lang_id, langs1=langs)

    return encoder_inputs


def convert_ids2grouped_bpe_tokens(ids, dico):
    sentence = []
    tokens = []
    for idx in ids:
        token = dico.id2word[idx]
        if "@@" not in token:
            tokens.append(token)
            sentence.append(tokens)
            tokens = []
        else:
            tokens.append(token)
    assert len(tokens) == 0
    return sentence


def show_alignments(src, tgt, dico, alignments):
    src = src.transpose(0, 1)
    tgt = tgt.transpose(0, 1)
    src_sentences = [convert_ids2grouped_bpe_tokens(list(ids.numpy()), dico) for ids in src]
    tgt_sentences = [convert_ids2grouped_bpe_tokens(list(ids.numpy()), dico) for ids in tgt]

    for src_batch_idx, src_length_idx, tgt_batch_idx, tgt_length_idx in zip(
            alignments.src_batch_size_index, alignments.src_length_index, alignments.tgt_batch_size_index, alignments.tgt_length_index):
        src_word = src_sentences[src_batch_idx][src_length_idx + 1]  # +1 because of eos
        tgt_word = tgt_sentences[tgt_batch_idx][tgt_length_idx + 1]  # +1 because of eos
        print("{}-{}".format(src_word, tgt_word))


def eval_alignment(combiner_seq2seq, dataset, lang2id, debug=False):
    type2dis = {alignment_type: [] for alignment_type in AlignmentTypes}
    type2num = {alignment_type: 0 for alignment_type in AlignmentTypes}

    for src, src_len, tgt, tgt_len, alignments in dataset.get_iterator():
        src_inputs = get_encoder_inputs(src, src_len, lang2id[dataset.src_lang])
        tgt_inputs = get_encoder_inputs(tgt, tgt_len, lang2id[dataset.tgt_lang])

        combiner_seq2seq.encoder.eval()
        with torch.no_grad():
            # forward
            src_encoded = combiner_seq2seq.encoder.encode(src_inputs).encoded
            tgt_encoded = combiner_seq2seq.encoder.encode(tgt_inputs).encoded

            # show alignments
            if debug:
                show_alignments(src, tgt, dataset.dico, alignments)

            # extract specific words rep
            dim = src_encoded.size(-1)
            src_representations = src_encoded[alignments.src_batch_size_index, alignments.src_length_index + 1].view(-1, dim) # +1 because of bos
            tgt_representations = tgt_encoded[alignments.tgt_batch_size_index, alignments.tgt_length_index + 1].view(-1, dim)
            sims = torch.nn.CosineSimilarity(dim=-1)(src_representations, tgt_representations)

            assert sims.size(0) == len(alignments.alignment_types)
            # update dis sum words sum
            for sim, alignment_type in zip(sims, alignments.alignment_types):
                type2dis[alignment_type].append(sim.item())
                type2num[alignment_type] += 1

    type2ave_dis = {}
    type2var = {}

    for alignment_type in AlignmentTypes:
        if type2num[alignment_type] == 0:
            type2ave_dis[alignment_type] = -1
        else:
            dis = np.array(type2dis[alignment_type])
            num = type2num[alignment_type]
            average_dis = dis.mean()
            var = dis.var()
            type2ave_dis[alignment_type] = average_dis
            type2var[alignment_type] = var

    return type2ave_dis, type2var, type2num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--bped_src", type=str)
    parser.add_argument("--bped_tgt", type=str)
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--alignments", type=str)
    parser.add_argument("--batch_size", type=int, default=32)

    # hack
    parser.add_argument("--average_hack", action="store_true", default=False)
    if parser.parse_known_args()[0].average_hack is True:
        parser.add_argument("--mass_checkpoint_for_hack", type=str, default="", help="also hack the encoder and decoder using a mass model")

    # debug
    parser.add_argument("--debug", action="store_true", default=False)

    eval_args = parser.parse_args()

    dico, train_params, combiner_seq2seq = load_combiner_model(model_path=eval_args.checkpoint)

    if eval_args.average_hack:
        combiner_seq2seq = hack(mass_checkpoint=eval_args.mass_checkpoint_for_hack, combiner_seq2seq=combiner_seq2seq)

    dataset = AlignmentDataset(
        src_bped_path=eval_args.bped_src,
        tgt_bped_path=eval_args.bped_tgt,
        alignment_path=eval_args.alignments,
        batch_size=eval_args.batch_size,
        dico=dico,
        params=train_params,
        src_lang=eval_args.src_lang,
        tgt_lang=eval_args.tgt_lang
    )

    type2ave_dis, type2var, type2num = eval_alignment(combiner_seq2seq=combiner_seq2seq, dataset=dataset, lang2id=train_params.lang2id)

    for alignment_type in AlignmentTypes:
        print("Type: {} Number: {} average dis: {} variance: {}".format(alignment_type, type2num[alignment_type], type2ave_dis[alignment_type], type2var[alignment_type]))

    print("Done")


if __name__ == "__main__":
    main()