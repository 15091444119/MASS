import torch

def combine_splitted_whole_representation(splitted_rep, combined_rep, combine_labels, length_after_combine):
    """
    For splitted word, we use the whole word representation.
    Params:
        splitted_rep: [bs, splitted_len, dim]
        combined_rep: [numer of combined words, dim]
        combine_labels: where to use whole word representation [bs, splitted_len]
        length_after_combine: [bs]
            length after combine for each sentence
    Returns:
        final_rep: [bs, length_after_combine
    """
    final_rep = torch.FloatTensor(length_after_combine.size()).cuda()

    final_rep_with_splitted_rep_mask, final_rep_with_combined_rep_mask, splitted_rep_for_final_rep_mask = \
        get_masks_for_combine_splitted_whole_rep(combie_labels=combine_labels, length_after_combine=length_after_combine)

    final_rep[final_rep_with_splitted_rep_mask] = splitted_rep[splitted_rep_for_final_rep_mask]

    final_rep[final_rep_with_combined_rep_mask] = combined_rep

    return final_rep

def get_masks_for_combine_splitted_whole_rep(combine_labels, length_after_combine):

    bs = combine_labels.size(0)
    final_rep_with_splitted_rep_mask = torch.Bool(length_after_combine.size()).fill_(False)
    final_rep_with_combined_rep_mask = torch.Bool(length_after_combine.size()).fill_(False)
    splitted_rep_for_final_rep_mask = torch.Bool(combine_labels.size()).fill_(False)

    for i in range(bs):















    final_rep_with_combined_rep_mask
