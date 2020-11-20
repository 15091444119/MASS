from .mass import set_model_mode, DecodingBatch
from ..combine_utils import CombineTool



class CommonCombineModel(object):

    def __init__(self, encoder, decoder, combiner):
        self.encoder = encoder
        self.decoder = decoder
        self.combiner = combiner


class CommonCombineStat(object):

    def __init__(self, batch):
        self.processed_s = batch.len2.size(0)
        self.processed_w = (batch.len2 - 1).sum().item()


class CommonCombineLosses(object):

    def __init__(self, decoding_loss):
        self.decoding_loss = decoding_loss


class CommonCombineBatch(object):

    def __init__(self, mass_batch, params):
        self.x1 = mass_batch.x1
        self.len1 = mass_batch.len1
        self.x2 = mass_batch.x2
        self.len2 = mass_batch.len2
        self.y = mass_batch.y
        self.pred_mask = mass_batch.pred_mask
        self.positions = mass_batch.positions
        self.lang_id = mass_batch.lang_id
        self.langs1 = mass_batch.langs1
        self.langs2 = mass_batch.langs2


def combiner_mass(models, mass_batch, params, dico, mode):
    # prepare
    set_model_mode(mode=mode, models=[models.encoder, models.combiner, models.decoder])

    batch = CommonCombineBatch(mass_batch=mass_batch, params=params)

    combine_tool = CombineTool(
        batch=batch.x1,
        length=batch.len1,
        dico=dico,
        mask_index=params.mask_index
    )

    # combine and maybe train combiner
    final_encoded = encode(common_combine_batch=batch,
                           combiner=models.combiner,
                           encoder=models.encoder,
                           combine_tool=combine_tool
    )

    # decoding
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

    losses = CommonCombineLosses(decoding_loss=decoding_loss)

    statistics = CommonCombineStat(batch=batch)

    return scores, losses, statistics


def encode(common_combine_batch, combiner, encoder, combine_tool):
    encoded = encoder(
        "fwd",
        x=common_combine_batch.x1,
        lengths=common_combine_batch.len1,
        langs=common_combine_batch.langs1,
        causal=False
    )

    combined_rep = combiner.combine(
        encoded=encoded,
        length=combine_tool.final_length,
        combine_labels=combine_tool.combine_labels,
        lang_id=common_combine_batch.lang_id
    )

    final_encoded = combine_tool.gather(splitted_rep=encoded, combined_rep=combined_rep)

    return final_encoded

