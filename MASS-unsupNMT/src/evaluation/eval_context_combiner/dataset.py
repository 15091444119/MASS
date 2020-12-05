import torch
import argparse
import numpy as np
from src.model.encoder import EncoderInputs
from src.evaluation.utils import load_combiner_model, to_cuda

AlignmentTypes = ["src_whole_tgt_sep", "src_whole_tgt_whole", "src_sep_tgt_sep", "src_sep_tgt_whole"]


class SentenceAlignment(object):

    def __init__(self):
        self.src_positions = []
        self.tgt_positions = []

    def add_position(self, src_position, tgt_position):
        self.src_positions.append(src_position)
        self.tgt_positions.append(tgt_position)


class AllTypeSentenceAlignment(object):

    def __init__(self, src_whole_tgt_sep: SentenceAlignment, src_whole_tgt_whole, src_sep_tgt_sep, src_sep_tgt_whole):
        self.src_whole_tgt_sep = src_whole_tgt_sep
        self.src_whole_tgt_whole = src_whole_tgt_whole
        self.src_sep_tgt_sep = src_sep_tgt_sep
        self.src_sep_tgt_whole = src_sep_tgt_whole


class BatchAlignment(object):

    def __init__(self, src_batch_size_index, src_length_index, tgt_batch_size_index, tgt_length_index):
        self.src_batch_size_index = src_batch_size_index
        self.src_length_index = src_length_index
        self.tgt_batch_size_index = tgt_batch_size_index
        self.tgt_length_index = tgt_length_index


class AllTypeBatchAlignment(object):
    """
    Batch alingment mask, for each alignment type, we have a source positions and a target positions,
    """

    def __init__(self, src_whole_tgt_sep: BatchAlignment, src_whole_tgt_whole, src_sep_tgt_sep, src_sep_tgt_whole):
        self.src_whole_tgt_sep = src_whole_tgt_sep
        self.src_whole_tgt_whole = src_whole_tgt_whole
        self.src_sep_tgt_sep = src_sep_tgt_sep
        self.src_sep_tgt_whole = src_sep_tgt_whole


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
        )

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
            positions: list of WholeSeparateAlignmentPositions
        """
        corpus_alignments = []

        for src_bped_sentence, tgt_bped_sentence, alignment in zip(src_bped_sentences, tgt_bped_sentences, string_alignments):
            src_whole_tgt_whole = SentenceAlignment()
            src_whole_tgt_sep = SentenceAlignment()
            src_sep_tgt_whole = SentenceAlignment()
            src_sep_tgt_sep = SentenceAlignment()

            for word_align in alignment.split(' '):
                src_index, tgt_index = word_align.split('-')
                len_src_word = len(src_bped_sentences[src_index])
                len_tgt_word = len(tgt_bped_sentences[tgt_index])
                if len_src_word == 1 and len_tgt_word == 1:
                    src_whole_tgt_whole.add_position(src_position=src_index, tgt_position=tgt_index)
                elif len_src_word == 1 and len_tgt_word > 1:
                    src_whole_tgt_sep.add_position(src_position=src_index, tgt_position=tgt_index)
                elif len_src_word > 1 and len_tgt_word == 1:
                    src_sep_tgt_whole.add_position(src_position=src_index, tgt_position=tgt_index)
                elif len_src_word > 1 and len_tgt_word > 1:
                    src_sep_tgt_sep.add_position(src_position=src_index, tgt_position=tgt_index)

            all_type_sentence_alignment = AllTypeSentenceAlignment(
                src_whole_tgt_sep=src_whole_tgt_sep,
                src_whole_tgt_whole=src_whole_tgt_whole,
                src_sep_tgt_sep=src_sep_tgt_sep,
                src_sep_tgt_whole=src_sep_tgt_whole
            )

            corpus_alignments.append(all_type_sentence_alignment)

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
                sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def batch_alignments(self, alignments):

        alignments_dict = {}

        for alignment_type in AlignmentTypes:
            src_batch_size_index = []
            src_length_index = []
            tgt_batch_size_index = []
            tgt_length_index = []

            for idx in range(len(alignments)):

                sentence_alignment = getattr(alignments[idx], alignment_type)

                for src_position in sentence_alignment.src_positions:
                    src_batch_size_index.append(idx)
                    src_length_index.append(src_position)
                for tgt_position in sentence_alignment.tgt_positions:
                    tgt_batch_size_index.append(idx)
                    tgt_length_index.append(tgt_position)

            batch_alignment = BatchAlignment(
                src_batch_size_index=torch.tensor(src_batch_size_index).long(),
                tgt_batch_size_index=torch.tensor(tgt_batch_size_index).long(),
                src_length_index=torch.tensor(src_length_index).long(),
                tgt_length_index=torch.tensor(tgt_length_index).long()
            )

            alignments_dict[alignment_type] = batch_alignment

        return AllTypeBatchAlignment(**alignments_dict)


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

        return iterator

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

    langs = x.clone().copy(lang_id)
    x, len, langs = to_cuda(x, len, langs)
    encoder_inputs = EncoderInputs(x1=x, len1=len, lang_id=lang_id, langs1=langs)

    return encoder_inputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--bped_src", type=str)
    parser.add_argument("--bped_tgt", type=str)
    parser.add_argument("--src_lang", type=str)
    parser.add_argument("--tgt_lang", type=str)
    parser.add_argument("--alignments", type=str)
    parser.add_argument("--batch_size", type=int, default=32)

    eval_args = parser.parse_args()

    dico, train_params, combiner_seq2seq = load_combiner_model(model_path=eval_args.checkpoint)

    dataset = AlignmentDataset(
        src_bped_path=eval_args.bped_src,
        tgt_bped_path=eval_args.bped_tgt,
        alignment_path=eval_args.alignments,
        batch_size=eval_args.batch_size,
        dico=dico,
        params=train_params
    )

    dis_sum = {alignment_type: 0 for alignment_type in AlignmentTypes}
    words_sum = {alignment_type: 0 for alignment_type in AlignmentTypes}

    for src, src_len, tgt, tgt_len, all_type_alignments in dataset.get_iterator():
        src_inputs = get_encoder_inputs(src, src_len, train_params.lang2id[eval_args.src_lang])
        tgt_inputs = get_encoder_inputs(tgt, tgt_len, train_params.lang2id[eval_args.tgt_lang])

        src_encoded = combiner_seq2seq.encoder.encode(src_inputs).encoded
        tgt_encoded = combiner_seq2seq.encoder.encode(tgt_inputs).encoded

        for alignment_type in AlignmentTypes:
            single_type_alignments = getattr(all_type_alignments, alignment_type)
            dim = src_encoded.size(-1)
            src_representations = src_encoded[single_type_alignments.src_batch_size_index, single_type_alignments.src_length_index].view(-1, dim)
            tgt_representations = tgt_encoded[single_type_alignments.tgt_batch_size_index, single_type_alignments.tgt_length_index].view(-1, dim)
            n_words = src_representations.size(0)

            dis = torch.nn.CosineSimilarity(src_representations, tgt_representations)

            dis_sum[alignment_type] += dis * n_words
            words_sum[alignment_type] += n_words

    for alignment_type in AlignmentTypes:
        print("Alignment type: {} Cos distance: {}".format(alignment_type, dis_sum[alignment_type] / words_sum[alignment_type]))

    print("Done")

