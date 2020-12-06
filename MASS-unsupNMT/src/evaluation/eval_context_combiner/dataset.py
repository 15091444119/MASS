import torch
import argparse
import numpy as np
import pdb
from src.model.encoder import EncoderInputs
from src.utils import to_cuda
from src.model import load_combiner_model
from enum import Enum
from src.combiner.combiner import AverageCombiner


class AlignmentTypes(Enum):
    src_whole_tgt_sep = 0
    src_whole_tgt_whole = 1
    src_sep_tgt_sep = 2
    src_sep_tgt_whole = 3


def judge_alignment_type(len_src_word, len_tgt_word):
    """

    Args:
        len_src_word: number of tokens source word have
        len_tgt_word:

    Returns:

    """
    if len_src_word == 1 and len_tgt_word == 1:
        return AlignmentTypes.src_whole_tgt_whole
    elif len_src_word == 1 and len_tgt_word > 1:
        return AlignmentTypes.src_whole_tgt_sep
    elif len_src_word > 1 and len_tgt_word == 1:
        return AlignmentTypes.src_sep_tgt_whole
    elif len_src_word > 1 and len_tgt_word > 1:
        return AlignmentTypes.src_sep_tgt_sep


class SentenceAlignment(object):

    def __init__(self):
        self.src_positions = []
        self.tgt_positions = []
        self.alignment_types = []

    def add_position(self, src_position, tgt_position, alignment_type):
        #TODO this positions don't consider bos and eos
        self.src_positions.append(src_position)
        self.tgt_positions.append(tgt_position)
        self.alignment_types.append(alignment_type)


class BatchAlignment(object):

    def __init__(self, src_batch_size_index, src_length_index, tgt_batch_size_index, tgt_length_index, alignment_types):
        """

        Args:
            src_batch_size_index: torch.long
                batch size index of each word
            src_length_index:  torch.long
                length index of each word
            tgt_batch_size_index:
            tgt_length_index:
            alignment_types: list
                list of alignment type
        """
        self.src_batch_size_index = src_batch_size_index
        self.src_length_index = src_length_index
        self.tgt_batch_size_index = tgt_batch_size_index
        self.tgt_length_index = tgt_length_index
        self.alignment_types = alignment_types


class AlignmentDataset(object):

    def __init__(self, src_bped_path, tgt_bped_path, alignment_path, batch_size, dico, params):
        self.batch_size = batch_size
        self.dico = dico
        self.pad_index = params.pad_index
        self.eos_index = params.eos_index
        src_bped_sentences = group_tokens(read_sentences(src_bped_path))
        tgt_bped_sentences = group_tokens(read_sentences(tgt_bped_path))
        string_alignments = read_alignments(alignment_path)

        src_bped_sentences, tgt_bped_sentences, string_alignments = self.filter_not_one2one_alignment(
            src_bped_sentences=src_bped_sentences,
            tgt_bped_sentences=tgt_bped_sentences,
            alignments=string_alignments
        )

        self.alignments = self.get_alignments(
            src_bped_sentences=src_bped_sentences,
            tgt_bped_sentences=tgt_bped_sentences,
            string_alignments=string_alignments
        )  # this alignments is on word level, not bped level

        self.src_indexed_sentences = self.index_sentences(src_bped_sentences, self.dico)
        self.tgt_indexed_sentences = self.index_sentences(tgt_bped_sentences, self.dico)

    def index_sentence(self, grouped_bped_sentence, dico):
        """
        Args:
            grouped_bped_sentence:
                like [["a"], ["b@@", "c"], ["d"]]
            dico:

        Returns:
            indexed_sentence:
                like [1, 2, 3, 4]
        """
        indexed_sentence = []
        for word in grouped_bped_sentence:
            for token in word:
                indexed_sentence.append(dico.index(token))
        return indexed_sentence

    def index_sentences(self, grouped_bped_sentences, dico):

        indexed_sentences = []
        for sentence in grouped_bped_sentences:
            indexed_sentences.append(self.index_sentence(grouped_bped_sentence=sentence, dico=dico))
        return indexed_sentences

    def filter_not_one2one_alignment(self, src_bped_sentences, tgt_bped_sentences, alignments):
        """
        Args:
            src_bped_sentences:
            tgt_bped_sentences:
            alignments:

        Returns:

        """
        final_src_bped_sentences = []
        final_tgt_bped_sentences = []
        final_alignments = []
        for src_bped_sentence, tgt_bped_sentence, alignment in zip(src_bped_sentences, tgt_bped_sentences, alignments):
            alignment = filter_alignment_one2one(alignment)
            if alignment == "":
                continue
            else:
                final_src_bped_sentences.append(src_bped_sentence)
                final_tgt_bped_sentences.append(tgt_bped_sentence)
                final_alignments.append(alignment)

        return final_src_bped_sentences, final_tgt_bped_sentences, final_alignments

    def get_alignments(self, src_bped_sentences, tgt_bped_sentences, string_alignments):
        """

        Args:
            src_bped_sentences:
            tgt_bped_sentences:
            alignments:

        Returns:
            positions: list of sentence_alignment
        """
        corpus_alignments = []

        for src_bped_sentence, tgt_bped_sentence, alignment in zip(src_bped_sentences, tgt_bped_sentences, string_alignments):
            sentence_alignment = SentenceAlignment()

            for word_align in alignment.split(' '):
                src_index, tgt_index = word_align.split('-')
                src_index, tgt_index = int(src_index), int(tgt_index)
                len_src_word = len(src_bped_sentence[src_index])
                len_tgt_word = len(tgt_bped_sentence[tgt_index])

                alignment_type = judge_alignment_type(len_src_word=len_src_word, len_tgt_word=len_tgt_word)

                sentence_alignment.add_position(
                    src_position=src_index,
                    tgt_position=tgt_index,
                    alignment_type=alignment_type
                )

            corpus_alignments.append(sentence_alignment)

        return corpus_alignments

    def batch_sentences(self, indexed_sentences):
        """
        batch a list of indexed sentences
        Args:
            indexed_sentences:

        Returns:
            batch
            length
        """
        lengths = torch.LongTensor([len(s) + 2 for s in indexed_sentences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)

        sent[0] = self.eos_index
        for i, s in enumerate(indexed_sentences):
            if lengths[i] > 2:  # if sentence not empty
                sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(np.array(s).astype(np.int64)))
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def batch_alignments(self, alignments):
        src_batch_size_index = []
        src_length_index = []
        tgt_batch_size_index = []
        tgt_length_index = []
        alignment_types = []

        for idx in range(len(alignments)):
            sentence_alignment = alignments[idx]
            for src_position in sentence_alignment.src_positions:
                src_batch_size_index.append(idx)
                src_length_index.append(src_position)
            for tgt_position in sentence_alignment.tgt_positions:
                tgt_batch_size_index.append(idx)
                tgt_length_index.append(tgt_position)
            for alignment_type in sentence_alignment.alignment_types:
                alignment_types.append(alignment_type)

        batch_alignment = BatchAlignment(
            src_batch_size_index=torch.tensor(src_batch_size_index).long(),
            tgt_batch_size_index=torch.tensor(tgt_batch_size_index).long(),
            src_length_index=torch.tensor(src_length_index).long(),
            tgt_length_index=torch.tensor(tgt_length_index).long(),
            alignment_types=alignment_types
        )

        return batch_alignment


    def get_batch(self, indices):
        src, src_len = self.batch_sentences([self.src_indexed_sentences[indice] for indice in indices])
        tgt, tgt_len = self.batch_sentences([self.tgt_indexed_sentences[indice] for indice in indices])
        alignments = self.batch_alignments([self.alignments[indice] for indice in indices])

        return src, src_len, tgt, tgt_len, alignments

    def get_iterator(self):

        def iterator():
            n_sentences = len(self.src_indexed_sentences)
            batch_size = self.batch_size
            for i in range(0, n_sentences, batch_size):
                j = min(i + self.batch_size, n_sentences)
                yield self.get_batch(list(range(i, j)))

        return iterator()


def filter_alignment_one2one(alignment):
    """ Delete one to many and many to one in the alignment
    Params:
        alignment: fastalign output, like "0-0 1-1 2-3"
    Returns:
        one-one alignment
    Example:
        alignment: "0-0 0-1 1-2 2-2 3-2 4-4"
        output: "4-4"
    """
    s2t = {}
    t2s = {}
    for word_align in alignment.rstrip().split(' '):
        src_id, tgt_id = word_align.split('-')
        if src_id not in s2t:
            s2t[src_id] = [tgt_id]
        else:
            s2t[src_id].append(tgt_id)
        if tgt_id not in t2s:
            t2s[tgt_id] = [src_id]
        else:
            t2s[tgt_id].append(src_id)

    filtered_alignment = []
    for src_id, tgt_id_list in s2t.items():
        if len(tgt_id_list) == 1:
            if len(t2s[tgt_id_list[0]]) == 1:
                filtered_alignment.append("{}-{}".format(src_id, tgt_id_list[0]))

    return ' '.join(filtered_alignment)


def read_alignments(alignment_path):
    """ Read alignments induced by fast-align """
    alignments = []
    with open(alignment_path, 'r') as f:
        for line in f:
            alignments.append(line.rstrip())
    return alignments


def read_sentences(path):
    """ Read sentences """
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            sentences.append(line.rstrip())
    return sentences


def group_tokens(bped_sentences):
    """ group tokens of one word in to a list
    For example
    He@@ llo , na@@ ncy ! will be saved as [["he@@", "llo"], [","], ["na@@", "cy"], ["!"]]
    """

    sentences = []
    for bped_sentence in bped_sentences:
        cur_word = []
        cur_sentence = []
        for token in bped_sentence.split(' '):
            cur_word.append(token)
            if not token.endswith("@@"):
                cur_sentence.append(cur_word)
                cur_word = []
        if len(cur_word) != 0: # the end of sentence is a bpe token, like "你@@ 好@@"
            cur_sentence.append(cur_word)
        sentences.append(cur_sentence)
    return sentences


def get_encoder_inputs(x, len, lang_id):

    langs = x.clone().fill_(lang_id)
    x, len, langs = to_cuda(x, len, langs)
    encoder_inputs = EncoderInputs(x1=x, len1=len, lang_id=lang_id, langs1=langs)

    return encoder_inputs


def hack_to_average_combiner(mass_checkpoint, combiner_seq2seq):
    """
    1.  use average combiner instead of the original combiner
    2. use the params in the mass checkpoint
    Args:
        mass_checkpoint:
        combiner_seq2seq:

    Returns:
    """
    print("HACK to Average combiner")
    reloaded = torch.load(mass_checkpoint)
    enc_reload = reloaded['encoder']
    if all([k.startswith('module.') for k in enc_reload.keys()]):
        enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}
    dec_reload = reloaded['decoder']
    if all([k.startswith('module.') for k in dec_reload.keys()]):
        dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}
    combiner_seq2seq.encoder.encoder.load_state_dict(enc_reload)
    combiner_seq2seq.decoder.load_state_dict(dec_reload)
    combiner_seq2seq.encoder.combiner = AverageCombiner()

    return combiner_seq2seq


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--bped_src", type=str)
    parser.add_argument("--bped_tgt", type=str)
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--alignments", type=str)
    parser.add_argument("--batch_size", type=int, default=32)

    # hack to an average combiner
    parser.add_argument("--average_hack", action="store_true", default=False)
    if parser.parse_known_args()[0].average_hack is True:
        parser.add_argument("--mass_checkpoint_for_hack", required=False)

    # debug
    parser.add_argument("--debug", action="store_true", default=False)

    eval_args = parser.parse_args()

    dico, train_params, combiner_seq2seq = load_combiner_model(model_path=eval_args.checkpoint)

    if eval_args.average_hack:
        combiner_seq2seq = hack_to_average_combiner(mass_checkpoint=eval_args.mass_checkpoint_for_hack, combiner_seq2seq=combiner_seq2seq)

    dataset = AlignmentDataset(
        src_bped_path=eval_args.bped_src,
        tgt_bped_path=eval_args.bped_tgt,
        alignment_path=eval_args.alignments,
        batch_size=eval_args.batch_size,
        dico=dico,
        params=train_params
    )

    type2dis = {alignment_type: [] for alignment_type in AlignmentTypes}
    type2num = {alignment_type: 0 for alignment_type in AlignmentTypes}

    for src, src_len, tgt, tgt_len, alignments in dataset.get_iterator():
        src_inputs = get_encoder_inputs(src, src_len, train_params.lang2id[eval_args.src_lang])
        tgt_inputs = get_encoder_inputs(tgt, tgt_len, train_params.lang2id[eval_args.tgt_lang])

        combiner_seq2seq.encoder.eval()
        with torch.no_grad():
            # forward
            src_encoded = combiner_seq2seq.encoder.encode(src_inputs).encoded
            tgt_encoded = combiner_seq2seq.encoder.encode(tgt_inputs).encoded

            # show alignments
            if eval_args.debug:
                show_alignments(src, tgt, dico, alignments)

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

    for alignment_type in AlignmentTypes:
        if type2num[alignment_type] == 0:
            print("Alignment type: {} No words".format(alignment_type))
        else:
            dis = np.array(type2dis[alignment_type])
            num = type2num[alignment_type]
            average_dis = dis.mean()
            var = dis.var()
            print("Alignment type: {} Average Cos distance: {}, Varience: {}, Number: {}".format(alignment_type, average_dis, var, num))

    print("Done")

