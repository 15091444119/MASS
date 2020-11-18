"""
cheat means we use original whole word representation as combined results
"""


from .mass import set_model_mode, DecodingBatch
from ..combine_utils import ExplicitSplitCombineTool
from src.combiner.forward_function.mass import MassBatch
from .explicit_split import ExplicitSplitCombineTool, ExplicitSplitBatch


class CheatModel(object):

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder


class Cheatstat(object):

    def __init__(self, batch, trained_words):
        self.processed_s = batch.len2.size(0)
        self.processed_w = (batch.len2 - 1).sum().item()


class CheatLosses(object):

    def __init__(self, decoding_loss):
        self.decoding_loss = decoding_loss



def combiner_mass_with_explict_split(models, mass_batch:MassBatch, params, dico, splitter, mode):
    set_model_mode(mode=mode, models=[models.encoder, models.combiner, models.decoder])

    batch = ExplicitSplitBatch(mass_batch=mass_batch, params=params, dico=dico, splitter=splitter)

    combine_tool = ExplicitSplitCombineTool(splitted_batch=batch.x3, length_before_split=batch.len1,
                                            length_after_split=batch.len3, dico=dico, mappers=batch.mappers)

    final_encoded = cheat_encode(cheat_batch=batch, encoder=models.encoder, combine_tool=combine_tool)

    decoding_batch = DecodingBatch(
        x=batch.x2,
        length=batch.len2,
        langs=batch.langs2,
        src_enc=final_encoded,
        src_len=combine_tool.final_length,
        src_mask=combine_tool.mask_for_decoder,
        positions=batch.positions,
        pred_mask=batch.pred_mask,
        y=batch.y
    )

    scores, decoding_loss = decoding_batch.decode(decoder=models.decoder, get_scores=(mode == "eval"))

    losses = CheatLosses(decoding_loss=decoding_loss)

    statistics = Cheatstat(batch=batch, trained_words=combine_tool.trained_combiner_words)

    return scores, losses, statistics


def cheat_encode(cheat_batch, encoder, combine_tool):
    orignal_encoded = encoder(
        "fwd",
        x=cheat_batch.x1,
        length=cheat_batch.len1,
        langs=cheat_batch.langs1,
        casual=False
    )

    splitted_encoded = encoder(
        "fwd",
        x=cheat_batch.x3,
        length=cheat_batch.len3,
        langs=cheat_batch.langs3,
        casual=False
    )

    # cheat!
    dim = orignal_encoded.size(-1)
    cheated_combine_rep = orignal_encoded.masked_select(combine_tool.splitted_original_word_mask).view(-1, dim)

    final_encoded = combine_tool.gather(splitted_rep=splitted_encoded, combined_rep=cheated_combine_rep)

    return final_encoded

