from .mass import MassBatch, set_model_mode, DecodingBatch
from ..combine_utils import ExplicitSplitCombineTool


def combiner_mass_with_explict_split(models, mass_batch:MassBatch, params, dico, splitter, mode, combine_loss_fn):
    """

    For monolingual data, we split whole word and get the encoder representation, and don't split whole word for another time,
    we let the whole word encoding representation to be similar to the combined whole word.

    Final loss have two parts:
        1. combiner loss (with combiner loss function like cos)
        2. mass loss (use the splitted input but not the whole word input)

    params:
        models: dict
            {"encoder":encoder,  "combiner":combiner, decoder:"decoder"}
        data: dict
            {"batch":batch, "length":length, "lang":lang}
        params: mass params
        dico: dictionary
        splitter: splitter
        mode:
    """

    # prepare
    set_model_mode(mode=mode, models=[models.encoder, models.combiner, models.decoder])

    batch = ExplicitSplitBatch(mass_batch=mass_batch, params=params, dico=dico, splitter=splitter)

    combine_tool = ExplicitSplitCombineTool(splitted_batch=batch.x3, length_before_split=batch.len1,
                                            length_after_split=batch.len3, dico=dico, mappers=batch.mappers, mask_index=params.mask_index)

    # combine and maybe train combiner
    combine_loss, final_encoded = encode_and_maybe_explicit_train_combiner(explicit_splitted_batch=batch, combiner=models.combiner, encoder=models.encoder, combine_tool=combine_tool, combine_loss_fn=combine_loss_fn)

    # decoding
    decoding_batch = DecodingBatch(x=batch.x2,
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

    losses = ExplicitSplitLosses(combine_loss=combine_loss, decoding_loss=decoding_loss)

    statistics = ExplicitSplitStat(explicit_batch=batch, combine_tool=combine_tool)

    return scores, losses, statistics


def encode_and_maybe_explicit_train_combiner(explicit_splitted_batch, combiner, encoder, combine_tool, combine_loss_fn):
    encoded = encoder(
        "fwd",
        x=explicit_splitted_batch.x3,
        lengths=explicit_splitted_batch.len3,
        langs=explicit_splitted_batch.langs3,
        causal=False
    )

    combined_rep = combiner.combine(
        encoded=encoded,
        final_len=combine_tool.final_length,
        combine_labels=combine_tool.combine_labels,
        lang_id=explicit_splitted_batch.lang_id
    )

    combine_loss = calculate_combine_loss(
        encoder=encoder,
        explicit_splitted_batch=explicit_splitted_batch,
        combined_rep=combined_rep,
        combine_tool=combine_tool,
        loss_fn=combine_loss_fn
    )

    final_encoded = combine_tool.gather(splitted_rep=encoded, combined_rep=combined_rep)

    return combine_loss, final_encoded


def calculate_combine_loss(encoder, explicit_splitted_batch, combined_rep, combine_tool, loss_fn):

    if combine_tool.trained_combiner_words != 0:
        original_rep = encoder("fwd",
                               x=explicit_splitted_batch.x1,
                               length=explicit_splitted_batch.len1,
                               langs=explicit_splitted_batch.langs1,
                               casual=False
                               )
        dim = original_rep.size(-1)

        trained_original_words_rep = original_rep.masked_select(combine_tool.splitted_original_word_mask).view(-1, dim)

        trained_combined_words_rep = combined_rep.masked_select(combine_tool.select_trained_rep_from_combined_rep).view(-1, dim)

        combine_loss = loss_fn(trained_original_words_rep, trained_combined_words_rep)
        return combine_loss

    else:
        return None


class ExplicitSplitModel(object):

    def __init__(self, encoder, decoder, combiner):
        self.encoder = encoder
        self.decoder = decoder
        self.combiner = combiner


class ExplicitSplitStat(object):

    def __init__(self, explicit_batch, combine_tool):
        self.processed_s = explicit_batch.len2.size(0)
        self.processed_w = (explicit_batch.len2 - 1).sum().item()
        self.trained_combiner_words = combine_tool.trained_combiner_words


class ExplicitSplitLosses(object):

    def __init__(self, combine_loss, decoding_loss):
        self.combine_loss = combine_loss
        self.decoding_loss = decoding_loss


class ExplicitSplitBatch(object):

    def __init__(self, mass_batch, params, dico, splitter):
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

        # split whole words to train combiner
        x3, len3, mappers = splitter.re_encode_batch_sentences(batch=self.x1, lengths=self.len1, dico=dico, re_encode_rate=params.re_encode_rate)
        langs3 = x3.clone().fill_(self.lang_id)

        self.x3 = x3
        self.len3 = len3
        self.langs3 = langs3
        self.mappers = mappers
