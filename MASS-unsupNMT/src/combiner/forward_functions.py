from src.utils import to_cuda
from src.combiner.combine_utils import CombineTool, CheatCombineTool, ExplicitSplitCombineTool
import torch




def mass(models, data: MassBatch, params, dico, mode):





def explicit_splitted_cheated_output_mass(models, data: MassBatch, params, dico, splitter, mode):
    """
    1. Encode two times:
        1. original input
        2. split whole words
    2. Use the whole words representation in 1 to substitute representation in 2.
    3. Decode based on representation in 2
    """
    encoder = models["encoder"]
    decoder = models["decoder"]

    # get inputs

    x1, len1, x2, len2, y, pred_mask, positions, lang = data.x1, data.len1, data.x2, data.len2, data.y, data.pred_mask, data.positions, data.lang
    lang_id = params.lang2id[lang]
    langs1 = x1.clone().fill_(lang_id)
    langs2 = x2.clone().fill_(lang_id)
    x1, len1, x2, len2, y, pred_mask, positions, langs1, langs2  = \
        to_cuda(x1, len1, x2, len2, y, pred_mask, positions, langs1, langs2)

    # split whole words to train combiner
    x3, len3, mappers = splitter.re_encode_batch_sentences(x1, len1, dico, params.re_encode_rate)
    langs3 = x3.clone().fill_(lang_id)
    x3, len3, langs3 = to_cuda(x3, len3, langs3)

    combine_tool = ExplicitSplitCombineTool(splitted_batch=x3, length_before_split=len1, length_after_split=len3, dico=dico, mappers=mappers)

    if mode == "train":
        encoder.train()
        decoder.train()
    elif mode == "eval":
        encoder.eval()
        decoder.eval()
    else:
        raise ValueError

    # get original representation
    origin_word_rep = encoder("fwd", x=x1, lengths=len1, langs=langs1, causal=False)  # [len, bs, dim]
    whole_word_rep =  combine_tool.get_trained_word_rep(origin_word_rep.transpose(0, 1))

    # get splited representation
    splitted_rep = encoder("fwd", x=x3, lengths=len3, langs=langs3, causal=False)  # [len, bs, dim]
    splitted_rep = splitted_rep.transpose(0, 1)  # [bs, len, dim]

    # combine them together



    # decode
    dec2 = decoder('fwd',
                   x=x2, lengths=len2, langs=langs2, causal=True,
                   src_enc=final_encoded, src_len=final_lens, positions=positions, enc_mask=enc_mask)

    word_scores, mass_loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=(mode == "eval"))
    losses = {"mass_loss": mass_loss}
    statistics = {"processed_s": len2.size(0), "processed_w": (len2 - 1).sum().item()}

    return losses, statistics


def combiner_mass(models, data: MassBatch, params, dico, mode):
    """
       params:
           models: dict
               {"encoder":encoder,  "combiner":combiner, decoder:"decoder"}
           data: dict
               {"batch", "lengths", "x2", "len2", "y", "pred_mask", "positions"}
           params: mass params
           dico: dictionary
           mode: str
            train or eval
    """
    encoder = models["encoder"]
    combiner = models["combiner"]
    decoder = models["decoder"]

    # mass mask
    x1, len1, x2, len2, y, pred_mask, positions, lang = data.x1, data.len1, data.x2, data.len2, data.y, data.pred_mask, data.positions, data.lang
    lang_id = params.lang2id[lang]
    langs1 = x1.clone().fill_(lang_id)
    langs2 = x2.clone().fill_(lang_id)
    all_combine_labels = get_combine_labels(x1, dico)


    x1, len1, x2, len2, y, pred_mask, positions, langs1, langs2, all_combine_labels = \
        to_cuda(x1, len1, x2, len2, y, pred_mask, positions, langs1, langs2, all_combine_labels)

    if mode == "eval":
        encoder.eval()
        decoder.eval()
        combiner.eval()
    elif mode == "train":
        encoder.train()
        decoder.train()
        combiner.train()
    else:
        raise ValueError


    # combiner whole word representation
    rep = encoder("fwd", x=x1, lengths=len1, langs=langs1, causal=False)  # [len, bs, dim]
    rep = rep.transpose(0, 1)  # [bs, len, dim]

    # combiner forward(only combiner is trained)
    final_encoded, final_lens, final_mask_mask, _ = combiner("encode", rep, len1,
                                                             all_combine_labels,
                                                             lang,
                                                             )  # [combine_word_num, dim]

    enc_mask = ~final_mask_mask

    dec2 = decoder('fwd',
                   x=x2, lengths=len2, langs=langs2, causal=True,
                   src_enc=final_encoded, src_len=final_lens, positions=positions, enc_mask=enc_mask)

    word_scores, mass_loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=(mode == "eval"))
    losses = {"mass_loss": mass_loss}
    statistics = {"processed_s": len2.size(0), "processed_w": (len2 - 1).sum().item()}

    return word_scores, losses, statistics



