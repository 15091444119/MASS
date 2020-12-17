import torch
import pdb
from src.model.combiner.context_combiner.constant import COMBINE_END, COMBINE_FRONT, NOT_COMBINE, PAD


class BaseCombineTool(object):
    def __init__(self, original_length, combine_labels, splitted_batch, mask_index):
        self.original_length = original_length
        self.combine_labels = combine_labels
        self.final_length = get_length_after_combine(self.combine_labels)
        self.mask_for_decoder = get_mask_for_decoder(splitted_batch=splitted_batch, combine_labels=self.combine_labels, final_length=self.final_length, mask_index=mask_index)
        self.final_rep_using_splitted_rep_mask, self.final_rep_using_combined_rep_mask, self.splitted_rep_for_final_rep_mask = \
            get_masks_for_combine_reps(combine_labels=self.combine_labels, final_length=self.final_length)

    def gather(self, splitted_rep, combined_rep):
        return gather_splitted_combine_representation(
            splitted_rep=splitted_rep,
            combined_rep=combined_rep,
            final_length=self.final_length,
            final_rep_using_splitted_rep_mask=self.final_rep_using_splitted_rep_mask,
            final_rep_using_combined_rep_mask=self.final_rep_using_combined_rep_mask,
            splitted_rep_for_final_rep_mask=self.splitted_rep_for_final_rep_mask
        )


class CombineTool(BaseCombineTool):
    """
    Some variable needed for combining
    """

    def __init__(self, batch, length, dico, mask_index):
        original_length = length
        combine_labels = get_combine_labels(batch, dico)
        super().__init__(
            original_length=original_length,
            combine_labels=combine_labels,
            splitted_batch=batch,
            mask_index=mask_index
        )


class CheatCombineTool(BaseCombineTool):

    def __init__(self, splitted_batch, length_before_split, length_after_split, dico, mappers, mask_index):
        combine_labels = get_new_splitted_combine_labels(mappers=mappers, length_before_split=length_before_split, length_after_split=length_after_split)
        super().__init__(
            original_length=length_before_split,
            combine_labels=combine_labels,
            splitted_batch=splitted_batch,
            mask_index=mask_index
        )
        self.length_before_split = length_before_split
        self.length_after_split = length_after_split

        self.splitted_original_word_mask = get_splitted_words_mask(mappers, length_before_split)


class ExplicitSplitCombineTool(BaseCombineTool):

    def __init__(self, splitted_batch, length_before_split, length_after_split, dico, mappers, mask_index):

        combine_labels = get_combine_labels(splitted_batch, dico)
        super().__init__(
            original_length=length_before_split,
            combine_labels=combine_labels,
            splitted_batch=splitted_batch,
            mask_index=mask_index
        )

        self.length_before_split = length_before_split
        self.length_after_split = length_after_split
        self.splitted_original_word_mask = get_splitted_words_mask(mappers, length_before_split)
        self.trained_combiner_words = self.splitted_original_word_mask.long().sum().item()

        self.select_trained_rep_from_combined_rep = get_mask_for_select_combined_rep(
            all_subword_labels=self.combine_labels,
            new_generated_subwords_labels=get_new_splitted_combine_labels(mappers=mappers,length_before_split=length_before_split,length_after_split=length_after_split)
        )

        try:
            assert len(self.select_trained_rep_from_combined_rep) == (self.splitted_original_word_mask == True).long().sum()
        except AssertionError:
            pdb.set_trace()


def get_mask_for_select_combined_rep(all_subword_labels, new_generated_subwords_labels):
    """
    We need to select trained representation from combined representation, so a mask is needed
    Args:
        combine_labels: [bs, length_after_split]
        new_generated_subwords_labels: [bs, length_after_split]
    Returns:
        selecting_tensor: shape:[n_trained_whole_words]
    """

    assert all_subword_labels.size() == new_generated_subwords_labels.size()
    bs, len = all_subword_labels.size()

    position_in_combined_rep = 0
    positions = []
    for sentence_id in range(bs):
        for token_id, token_label in enumerate(all_subword_labels[sentence_id]):
            if token_label == COMBINE_END:
                if new_generated_subwords_labels[sentence_id][token_id] == COMBINE_END:
                    positions.append(position_in_combined_rep)
                position_in_combined_rep += 1

    return torch.tensor(positions).to(all_subword_labels.device)


def get_mask_for_decoder(splitted_batch, combine_labels, final_length, mask_index):
    """
    Get encoder mask for decoding, only masked token will be set to False, pad index is not considered
    Args:
        splitted_batch: [length_after_split, bs]
        combine_labels: [bs, length_after_split]
        final_length: [bs]
        mask_index:

    Returns:
        mask: [bs, final_length]
    """
    splitted_batch = splitted_batch.transpose(0, 1)  # [bs, length_after_combine]
    bs, max_length_after_split = splitted_batch.size()
    max_final_length = max(final_length).item()
    mask = torch.BoolTensor(bs, max_final_length).fill_(True)

    for i in range(bs):
        index_in_final_encoded = -1
        for idx, token_idx in enumerate(splitted_batch[i]):

            # update index in final encoded
            if combine_labels[i][idx] == PAD:
                break
            elif combine_labels[i][idx] == COMBINE_END or combine_labels[i][idx] == NOT_COMBINE:
                index_in_final_encoded += 1

            # set mask to false
            if token_idx.item() == mask_index:
                try:
                    assert combine_labels[i][idx] == NOT_COMBINE
                except:
                    pdb.set_trace()
                mask[i][index_in_final_encoded] = False

    return mask.to(splitted_batch.device)


def get_length_after_combine(combine_labels):
    length = combine_labels.eq(NOT_COMBINE).sum(dim=-1) + combine_labels.eq(COMBINE_END).sum(dim=-1)
    return length.to(combine_labels.device)


def get_combine_labels(batch, dico):
    """
    pad: -1
    whole_words(skipped token, containing masked token, eos, bos ...): 1
    subword front: 2
    subword end: 3

    params:
        batch: [len, bs]
        dico:

    returns:
        combine_labels: [bs, len]
    """
    seq_len, bs = batch.size()
    combine_labels = torch.LongTensor(bs, seq_len).fill_(PAD)

    for i in range(bs):
        word_finished = True
        for j in range(seq_len):
            cur_word = dico.id2word[batch[j][i].item()]
            label = 0

            if cur_word == dico.id2word[dico.pad_index]:
                label = PAD

            elif "@@" not in cur_word:
                if word_finished is False:
                    label = COMBINE_END
                else:
                    label = NOT_COMBINE
                word_finished = True

            elif "@@" in cur_word:
                label = COMBINE_FRONT
                word_finished = False

            combine_labels[i][j] = label

        assert word_finished
    return combine_labels.to(batch.device)


def get_splitted_words_mask(mappers, length_before_split):
    """
    trained whole word will be masked as True
    Params:
        mappers:
        lengths_before_split:
        new_lengths:
    Returns:
        trained_whole_word_mask: shape: (bs, len)
    """
    mask = torch.BoolTensor(length_before_split.size(0), length_before_split.max().item()).fill_(False)
    for sentence_id, mapper in enumerate(mappers):
        for idx in range(length_before_split[sentence_id].item()):
            if mapper[idx][1] - mapper[idx][0] > 1:  # is a word be separated
                mask[sentence_id][idx] = True
    return mask.to(length_before_split.device)


def get_new_splitted_combine_labels(mappers, length_before_split, length_after_split):
    """

     words didn't be new splitted will not count

    Params:
        mappers:
        length
        new_lengths:
    Returns:
        combine_labels: [bs, new_length]
    """
    combine_labels = torch.LongTensor(length_after_split.size(0), length_after_split.max().item()).fill_(PAD)

    for sentence_id, mapper in enumerate(mappers):
        for idx in range(length_before_split[sentence_id].item()):
            if mapper[idx][1] - mapper[idx][0] > 1:
                combine_labels[sentence_id][mapper[idx][0]:mapper[idx][1] - 1].fill_(COMBINE_FRONT)
                combine_labels[sentence_id][mapper[idx][1] - 1].fill_(COMBINE_END)
            else:
                combine_labels[sentence_id][mapper[idx][0]].fill_(NOT_COMBINE)

    return combine_labels.to(length_before_split.device)


def gather_splitted_combine_representation(splitted_rep, combined_rep, final_length, final_rep_using_splitted_rep_mask, final_rep_using_combined_rep_mask, splitted_rep_for_final_rep_mask):
    """

    Args:
        splitted_rep: [bs, splitted_len, dim]
        combined_rep: [combined_words, dim]
        final_length:  [bs]
        final_rep_using_splitted_rep_mask: [bs, final_len]
        final_rep_using_combined_rep_mask: [bs, final_len]
        splitted_rep_for_final_rep_mask: [bs, splitted_len]

    Returns:
        final_rep: [bs, final_len, dim]

    """
    if combined_rep is None:
        return splitted_rep

    bs = final_length.size(0)
    max_len_after_combine = max(final_length).item()
    dim = splitted_rep.size(-1)
    final_rep = torch.FloatTensor(bs, max_len_after_combine, dim).fill_(0).to(splitted_rep.device)

    final_rep[final_rep_using_splitted_rep_mask.unsqueeze(-1).expand_as(final_rep)] = splitted_rep[splitted_rep_for_final_rep_mask.unsqueeze(-1).expand_as(splitted_rep)]

    final_rep[final_rep_using_combined_rep_mask.unsqueeze(-1).expand_as(final_rep)] = combined_rep.view(-1)

    return final_rep


def get_masks_for_combine_reps(combine_labels, final_length):
    """
    params:
        combine_labels:  [bs, len]
            combine front -> combine end: using combined representation at last
            not combine: using the original representation
        length_after_combine: [bs]
    """

    bs = combine_labels.size(0)
    max_final_length = max(final_length)

    final_rep_using_splitted_rep_mask = torch.BoolTensor(bs, max_final_length).fill_(False).to(combine_labels.device)
    final_rep_using_combined_rep_mask = torch.BoolTensor(bs, max_final_length).fill_(False).to(combine_labels.device)

    splitted_rep_for_final_rep_mask = torch.eq(combine_labels, NOT_COMBINE).to(combine_labels.device)

    for sentence_id in range(bs):
        word_id_in_final_rep = 0
        for word_id_in_splitted_rep, label in enumerate(combine_labels[sentence_id]):
            if label == PAD:
                break
            elif label == COMBINE_END:
                final_rep_using_combined_rep_mask[sentence_id][word_id_in_final_rep] = True
                word_id_in_final_rep += 1
            elif label == NOT_COMBINE:
                final_rep_using_splitted_rep_mask[sentence_id][word_id_in_final_rep] = True
                word_id_in_final_rep += 1
            elif label == COMBINE_FRONT:
                continue
            else:
                raise ValueError

    return final_rep_using_splitted_rep_mask, final_rep_using_combined_rep_mask, splitted_rep_for_final_rep_mask
