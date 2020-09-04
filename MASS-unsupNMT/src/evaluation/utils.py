import torch
import logging
import torch.nn as nn
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel
from src.utils import AttrDict
from collections import OrderedDict
from src.utils import to_cuda

logger = logging.getLogger()

def load_mass_model(model_path):
    """ reload a mass model
    Params:
        model_path: path of mass model
    Returns:
        dico, model_params, encoder, decoder
    """
    reloaded = torch.load(model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=True).cuda().eval()
    decoder = TransformerModel(model_params, dico, is_encoder=False, with_output=True).cuda().eval()

    def package_module(modules):
        state_dict = OrderedDict()
        for k, v in modules.items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
        return state_dict

    encoder.load_state_dict(package_module(reloaded['encoder']))
    decoder.load_state_dict(package_module(reloaded['decoder']))

    return dico, model_params, encoder, decoder

def prepare_batch_input(sents, lang, dico, mass_params):
    word_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                        for s in sents]
    return word_ids2batch(word_ids, lang, mass_params)

def word_ids2batch(word_ids, lang, mass_params):
    """
    Params:
        word_ids: list of tensor containing word ids(no special tokens)
        lang: input langauge
        mass_params: params of mass model
    """
    lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
    batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(mass_params.pad_index)
    batch[0] = mass_params.eos_index
    for j, s in enumerate(word_ids):
        if lengths[j] > 2:  # if sentence not empty
            batch[1:lengths[j] - 1, j].copy_(s)
        batch[lengths[j] - 1, j] = mass_params.eos_index
    langs = batch.clone().fill_(mass_params.lang2id[lang])
    return batch, lengths, langs

def get_token_embedding(model, dico, token):
    """ Get embedding of the given toke 
    Params:
        model: transformer encoder or decoder
        dico: dictionary object
        token: token to get embedding
    Returns:
        a numpy array of token embedding
    """
    assert token in dico.word2id

    idx = dico.word2id[token]

    return model.embeddings.weight.data[idx]


def encode_tokens(encoder, dico, params, tokens, lang):
    """ get encoder output of the word
    Params:
        encoder: transformer encoder
        dico: dictionary object
        params: mass_params
        word: a list of tokens which consist the word like ["对@@", "不起"]
        lang_id: id of the language 
    Returns:
        Context representation of the given word, a torch tensor[len(word) + 2, dim]
        eos representation are in the left most and right most place
    """

    batch, lengths, langs = prepare_batch_input([' '.join(tokens)], lang, dico, params)
    if torch.cuda.is_available():
        batch, lengths, langs = to_cuda(batch, lengths, langs)

    with torch.no_grad():
        encoded = encoder.fwd(x=batch, lengths=lengths, langs=langs, causal=False)
        encoded = encoded.transpose(0, 1).squeeze() # [sequence_length, dim]
        
    return encoded


def encode_sentences(encoder, dico, mass_params, sentences, lang):
    """
    Encoder a batch of sentences
    Params:
        encoder:
        dico:
        params:
        sentences(list): list of strings, each string is a sentence(will be tokenized into tokens by .strip().split(), see prepare_batch_input()
        lang(string): language of the sentences
    Returns:
        encoded(batch_size, max_length, dim):
    """
    batch, lengths, langs = prepare_batch_input(sentences, lang, dico, mass_params)
    if torch.cuda.is_available():
        batch, lengths, langs = to_cuda(batch, lengths, langs)

    encoded = encoder.fwd(x=batch, lengths=lengths, langs=langs, causal=False)
    encoded = encoded.transpose(0, 1)  # [batch_size, sequence_length, dim]

    return encoded, lengths


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
            context.masked_fill(~mask, 0)
            context_sum = context.sum(dim=1)
            context_average = context_sum / lengths.unsqueeze(-1)
            return context_average
        elif self._method == "max_pool":
            mask = get_mask(lengths, True, expand=hidden_dim, batch_first=True, cuda=context.is_cuda)
            context_max_poll, _ = torch.max(context.masked_fill(~mask, -1e9), dim=1)
            return context_max_poll
