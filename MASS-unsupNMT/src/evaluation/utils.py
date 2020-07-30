import torch
import logging
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel
from src.utils import AttrDict
from collections import OrderedDict

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

def prepare_batch_input(sents, lang_id, dico, mass_params):
    """ prepare input for mass 
    params:
        sents: list of strings, each string is a sentence
        lang_id: language id of the input sentences
        dico: dictionary object
        mass_params: params reloaded from mass model
    returns:
        batch, lengths, langs
    """
    token_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                        for s in sents]
    lengths = torch.LongTensor([len(s) + 2 for s in token_ids])
    batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(mass_params.pad_index)
    batch[0] = mass_params.eos_index
    for j, s in enumerate(token_ids):
        if lengths[j] > 2:  # if sentence not empty
            batch[1:lengths[j] - 1, j].copy_(s)
        batch[lengths[j] - 1, j] = mass_params.eos_index
    langs = batch.clone().fill_(lang_id)
    return batch, lengths, langs

def encode_tokens(encoder, dico, params, tokens, lang_id):
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

    batch, lengths, langs = prepare_batch_input([' '.join(tokens)], lang_id, dico, params)

    with torch.no_grad():
        encoded = encoder.fwd(batch, lengths=lengths, langs=langs)
        encoded = encoded.transpose(0, 1).squeeze() # [sequence_length, dim]
        
    return encoded