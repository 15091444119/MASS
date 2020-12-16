import torch.nn as nn
import torch


class Context2Sentence(nn.Module):
    def __init__(self, context_extractor):
        super().__init__()
        self._method = context_extractor

    def forward(self, context, lengths):
        batch_size, seq_len, hidden_dim = context.size()
        if self._method == "last_time":
            mask = get_mask(lengths, False, expand=hidden_dim, batch_first=True, cuda=context.is_cuda)
            h_t = context.masked_select(mask).view(batch_size, hidden_dim)
            return h_t
        elif self._method == "average":
            mask = get_mask(lengths, True, expand=hidden_dim, batch_first=True, cuda=context.is_cuda)
            context = context.masked_fill(~mask, 0)
            context_sum = context.sum(dim=1)
            context_average = context_sum / lengths.unsqueeze(-1)
            return context_average
        elif self._method == "max_pool":
            mask = get_mask(lengths, True, expand=hidden_dim, batch_first=True, cuda=context.is_cuda)
            context_max_pool, _ = torch.max(context.masked_fill(~mask, -1e9), dim=1)
            return context_max_pool
        elif self._method == "before_eos":
            mask = before_eos_mask(lengths, hidden_dim=hidden_dim, cuda=context.is_cuda)
            h_t = context.masked_select(mask).view(batch_size, hidden_dim)
            return h_t
        elif self._method == "word_token_average":
            mask = word_token_mask(lengths, hidden_dim)
            context = context.masked_fill(~mask, 0)
            context_sum = context.sum(dim=1)
            context_average = context_sum / (lengths - 2).unsqueeze(-1)  # don't count bos and eos length
            return context_average


def get_mask(lengths, all_words, expand=None, ignore_first=False, batch_first=False, cuda=True):
    """
    Create a mask of shape (slen, bs) or (bs, slen).
    """
    bs, slen = lengths.size(0), lengths.max()
    mask = torch.BoolTensor(slen, bs).zero_()
    for i in range(bs):
        if all_words:
            mask[:lengths[i], i] = 1
        else:
            mask[lengths[i] - 1, i] = 1
    if expand is not None:
        assert type(expand) is int
        mask = mask.unsqueeze(2).expand(slen, bs, expand)
    if ignore_first:
        mask[0].fill_(0)
    if batch_first:
        mask = mask.transpose(0, 1)
    if cuda:
        mask = mask.cuda()
    return mask


def before_eos_mask(lengths, hidden_dim, cuda):
    """ mask the position before eos ( for whole word """

    bs, slen = lengths.size(0), lengths.max()
    mask = torch.BoolTensor(slen, bs).zero_()
    for i in range(bs):
        mask[lengths[i] - 2, i] = 1
    mask = mask.unsqueeze(2).expand(slen, bs, hidden_dim)
    mask = mask.transpose(0, 1)
    if cuda:
        mask = mask.cuda()
    return mask


def word_token_mask(lengths, hidden_dim):
    """
    don't mask bos, eos  and pad
    Args:
        lengths:
        hidden_dim:

    Returns:

    """
    bs, slen = lengths.size(0), lengths.max()
    mask = torch.BoolTensor(slen, bs).zero_()
    for i in range(bs):
        mask[1:lengths[i] - 1, i] = 1
    mask = mask.unsqueeze(2).expand(slen, bs, hidden_dim)
    mask = mask.transpose(0, 1)
    mask = mask.cuda()
    return mask
