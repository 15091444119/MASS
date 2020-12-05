import torch
import logging
import torch.nn as nn
import pdb
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel
from src.utils import AttrDict
from collections import OrderedDict
from src.utils import to_cuda
from src.model import build_model

logger = logging.getLogger()


def package_module(modules):
    """
    Return model state dict, take multi-gpu case into account
    """
    state_dict = OrderedDict()
    for k, v in modules.items():
        if k.startswith('module.'):
            state_dict[k[7:]] = v
        else:
            state_dict[k] = v
    return state_dict


def load_combiner_model(model_path):
    print("Load model from {}".format(model_path))
    reloaded = torch.load(model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])

    assert model_params.encoder == "combiner"
    combiner_seq2seq = build_model(model_params, dico)

    combiner_seq2seq.load_state_dict(package_module(reloaded["seq2seq_model"]))

    return dico, model_params, combiner_seq2seq

def load_mass_model(model_path):
    """ reload a mass model
    Params:
        model_path: path of mass model
    Returns:
        dico, model_params, encoder, decoder
    """
    print("Load model from {}".format(model_path))
    reloaded = torch.load(model_path)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=True).cuda().eval()
    decoder = TransformerModel(model_params, dico, is_encoder=False, with_output=True).cuda().eval()

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
        encoded = encoded.transpose(0, 1).squeeze()  # [sequence_length, dim]

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


def word_token_mask(lengths, hidden_dim):
    bs, slen = lengths.size(0), lengths.max()
    mask = torch.BoolTensor(slen, bs).zero_()
    for i in range(bs):
        mask[1:lengths[i] - 1, i] = 1
    mask = mask.unsqueeze(2).expand(slen, bs, hidden_dim)
    mask = mask.transpose(0, 1)
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
            context = context.masked_fill(~mask, 0)
            context_sum = context.sum(dim=1)
            context_average = context_sum / lengths.unsqueeze(-1)
            return context_average
        elif self._method == "max_pool":
            mask = get_mask(lengths, True, expand=hidden_dim, batch_first=True, cuda=context.is_cuda)
            context_max_poll, _ = torch.max(context.masked_fill(~mask, -1e9), dim=1)
            return context_max_poll
        elif self._method == "before_eos":
            mask = before_eos_mask(lengths, hidden_dim=hidden_dim, cuda=context.is_cuda)
            h_t = context.masked_select(mask).view(batch_size, hidden_dim)
            return h_t
        elif self._method == "word_token_average":
            mask = word_token_mask(lengths, hidden_dim)
            context = context.masked_fill(~mask, 0)
            context_sum = context.sum(dim=1)
            context_average = context_sum / (lengths - 2).unsqueeze(-1)
            return context_average


class SenteceEmbedder(nn.Module):
    """
    encode sentences into a single representation
    """

    def __init__(self, encoder, mass_params, dico, context_extractor):
        super().__init__()
        self._encoder = encoder
        self._mass_params = mass_params
        self._dico = dico
        self._context2sentence = Context2Sentence(context_extractor)

    def forward(self, sentences, lang):
        batch_context_word_representations, lengths = encode_sentences(self._encoder, self._dico, self._mass_params,
                                                                       sentences, lang)
        batch_sentence_representation = self._context2sentence(batch_context_word_representations, lengths)
        return batch_sentence_representation


class WordEmbedderWithCombiner(nn.Module):
    """
    encode word with mass encoder and a combiner
    """
    def __init__(self, encoder, combiner, mass_params, dico):
        super().__init__()
        self._encoder = encoder
        self._combiner = combiner
        self._mass_params = mass_params
        self._dico = dico

    def forward(self, sentences, lang):
        """
        Params:
            sentences: each sentence is just a word, tokenized by bpe

        Returns:
            batch_sentence_representation: torch.FloatTensor size:(batch_size, emb_dim)
        """
        batch_context_word_representations, lengths = encode_sentences(self._encoder, self._dico, self._mass_params,
                                                                       sentences, lang)


        batch_sentence_representation = self._combiner(batch_context_word_representations.transpose(0, 1), lengths, lang)

        return batch_sentence_representation

    def with_special_token_forward(self, sentences, lang):
        """
        Params:
            sentences: each sentence is just a word, tokenized by bpe

        Returns:
            batch_sentence_representation: torch.FloatTensor size:(3, batch_size, emb_dim)
                sentence representation of each word
        """
        batch_context_word_representations, lengths = encode_sentences(self._encoder, self._dico, self._mass_params,
                                                                       sentences, lang)

        for length in lengths:
            assert length.item() > 3 #(all are separated word)

        batch_sentence_representation = self._combiner(batch_context_word_representations.transpose(0, 1), lengths, lang)

        batch_size, emb_dim = batch_sentence_representation.size()
        outputs = torch.FloatTensor(batch_size, 3, emb_dim).to(batch_sentence_representation.device)
        for i in range(batch_size):
            outputs[i][1] = batch_sentence_representation[i]
            outputs[i][0] = batch_context_word_representations[i][0]
            outputs[i][2] = batch_context_word_representations[i][2]

        outputs_lengths = torch.tensor([3] * batch_size).to(batch_sentence_representation.device)
        return outputs.transpose(0, 1), outputs_lengths

