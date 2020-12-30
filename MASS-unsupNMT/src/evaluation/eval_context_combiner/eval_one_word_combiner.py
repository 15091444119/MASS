"""
对于一句话，要得到这句话中每个词的表示
1. encode一次，得到表示
2. 对每个被切分的词，都进行合成（不一样的combine label），得到每个被切分词的表示
最终得到每个被切分词的表示。
"""
import argparse
import torch
import numpy as np
from src.utils import to_cuda
from .dataset import AlignmentDataset, AlignmentTypes
from src.context_combiner.model import CombineTool
from src.context_combiner.model.context_combiner import AverageCombiner
from src.context_combiner.model.constant import PAD, COMBINE_END, COMBINE_FRONT, NOT_COMBINE
from src.evaluation.utils import load_mass_model
from src.utils import AttrDict


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


def build_metric_fn(metric):
    if metric == "COS":
        return torch.nn.CosineSimilarity(dim=-1)
    elif metric == "MSE":
        def mse(x, y):
            return torch.nn.MSELoss(reduction="none")(x, y).mean(dim=-1)
        return mse


def encode(combiner, encoder, lang_id, x, x_len, dico, mask_index):

    langs = x.clone().fill_(lang_id)
    x, x_len, langs = to_cuda(x, x_len, langs)

    encoded = encoder(
        "fwd",
        x=x,
        lengths=x_len,
        langs=langs,
        causal=False
    ).transpose(0, 1)  # [bs, len, dim]

    bs, max_len, dim = encoded.size()

    combine_tool = CombineTool(x, x_len, dico, mask_index)
    inputs = []
    lengths = []
    combine_labels = []

    # encode one time for one word
    for i in range(bs):
        print(i)
        combine_label = torch.tensor([PAD for i in range(max_len)])
        combine_label[0:x_len[i].item()] = NOT_COMBINE
        start = 0
        while(start < x_len[i].item()):
            end = start
            while(end < x_len[i].item() and "@@" in dico.id2word[x[end][i].item()]):
                end += 1
            if end == x_len[i].item():
                break
            if end - start >= 1:
                inputs.append(encoded[i])
                lengths.append(x_len[i].item())
                cur_label = combine_label.clone()
                cur_label[start:end] = COMBINE_FRONT
                cur_label[end] = COMBINE_END
                combine_labels.append(cur_label)
            start = end + 1
    inputs = torch.stack(inputs, dim=0).cuda()
    lengths = torch.tensor(lengths).cuda().long()
    combine_labels = torch.stack(combine_labels, dim=0).cuda().long()

    # change length
    max_length = lengths.max().item()
    inputs = inputs[:, :max_length]
    combine_labels = combine_labels[:, :max_length]

    reps = combiner.combine(inputs, lengths, combine_labels)

    final_reps = combine_tool.gather(encoded, reps)

    return final_reps


def eval_alignment(src_combiner, tgt_combiner, encoder, dataset, lang2id, mask_index, debug=False, metric="COS"):

    metric_fn = build_metric_fn(metric)

    src_combiner.eval()
    tgt_combiner.eval()
    encoder.eval()

    type2dis = {alignment_type: [] for alignment_type in AlignmentTypes}
    type2num = {alignment_type: 0 for alignment_type in AlignmentTypes}

    for src, src_len, tgt, tgt_len, alignments in dataset.get_iterator():
        with torch.no_grad():
            # forward
            # show alignments
            if debug:
                show_alignments(src, tgt, dataset.dico, alignments)

            # extract specific words rep
            src_encoded = encode(combiner=src_combiner, encoder=encoder, lang_id=lang2id[dataset.src_lang], x=src, x_len=src_len, dico=dataset.dico, mask_index=mask_index)
            tgt_encoded = encode(combiner=tgt_combiner, encoder=encoder, lang_id=lang2id[dataset.tgt_lang], x=tgt, x_len=tgt_len, dico=dataset.dico, mask_index=mask_index)

            dim = src_encoded.size(-1)
            src_representations = src_encoded[alignments.src_batch_size_index, alignments.src_length_index + 1].view(-1, dim) # +1 because of bos
            tgt_representations = tgt_encoded[alignments.tgt_batch_size_index, alignments.tgt_length_index + 1].view(-1, dim)
            sims = metric_fn(src_representations, tgt_representations)

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


def load_combiner(path):
    model = torch.load(path)

    params = model["params"]
    params = AttrDict(params)

    combiner = AverageCombiner().cuda()
    """
    combiner = build_combiner(params).cuda()

    combiner.load_state_dict(model["combiner"])
    """

    return combiner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_combiner", type=str)
    parser.add_argument("--tgt_combiner", type=str)
    parser.add_argument("--mass_model", type=str)
    parser.add_argument("--bped_src", type=str)
    parser.add_argument("--bped_tgt", type=str)
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--alignments", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--metric", choices=["MSE", "COS"], default="COS")

    # debug
    parser.add_argument("--debug", action="store_true", default=False)

    eval_args = parser.parse_args()

    dico, mass_params, encoder, decoder = load_mass_model(eval_args.mass_model)
    encoder = encoder.cuda()
    src_combiner = load_combiner(eval_args.src_combiner)
    tgt_combiner = load_combiner(eval_args.tgt_combiner)

    dataset = AlignmentDataset(
        src_bped_path=eval_args.bped_src,
        tgt_bped_path=eval_args.bped_tgt,
        alignment_path=eval_args.alignments,
        batch_size=eval_args.batch_size,
        dico=dico,
        src_lang=eval_args.src_lang,
        tgt_lang=eval_args.tgt_lang,
        pad_index=mass_params.pad_index,
        eos_index=mass_params.eos_index
    )

    type2ave_dis, type2var, type2num = eval_alignment(src_combiner=src_combiner, tgt_combiner=tgt_combiner, encoder=encoder, dataset=dataset, lang2id=mass_params.lang2id, metric=eval_args.metric, mask_index=mass_params.mask_index)

    for alignment_type in AlignmentTypes:
        print("Type: {} Number: {} average dis: {} variance: {}".format(alignment_type, type2num[alignment_type], type2ave_dis[alignment_type], type2var[alignment_type]))

    print("Done")


if __name__ == "__main__":
    main()