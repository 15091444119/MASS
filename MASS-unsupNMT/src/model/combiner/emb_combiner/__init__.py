from .emb_combiner import LinearEmbCombiner, GRUEmbCombiner, TransformerCombiner


def build_emb_combiner_model(emb_dim, params):
    """

    Args:
        emb_dim:
        params: params from train_emb_combiner.py

    Returns:

    """

    if params.combiner_type == "linear":
        combiner = LinearEmbCombiner(emb_dim=emb_dim, context_extractor=params.context_extractor_type)
    elif params.combiner_type == "gru":
        combiner = GRUEmbCombiner(emb_dim=emb_dim, n_layer=params.n_combiner_layer, context_extractor=params.context_extractor_type)
    elif params.combiner_type == "transformer":
        combiner = TransformerCombiner(emb_dim=emb_dim, n_layer=params.n_combiner_layer, n_head=params.n_head, context_extractor=params.context_extractor_type)
    else:
        raise NotImplementedError

    return combiner


