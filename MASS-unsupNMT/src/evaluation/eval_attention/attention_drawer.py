def draw_multi_layer_multi_head_attention(self, src_tokens, tgt_tokens, attention_weights, method, output_prefix):
    if method == "all":
        self._all_attention(src_tokens, tgt_tokens, attention_weights, output_prefix)
    elif method == "all_average"
        self._draw_all_average_attention(src_tokens, tgt_tokens, attention_weights, output_prefix)

def _draw_all_average_attention(self, src_tokens, tgt_tokens, attention_weights, output_prefix):
    output_path = output_prefix + "_all-average.jpg"
    weights_sum = np.zeros(len(tgt_tokens), len(src_tokens))
    for layer_id in range(attention_weights.n_layers):
        for head_id in range(attention_weights.n_heads):
            weights_sum += attention_weights.get_attention(sentence_id=0, layer_id=layer_id, head_id=head_id).cpu().numpy()
    weights_average = weights_sum / (attention_weights.n_layers * attention_weights.n_heads)
    draw_attention(weights_average, src_tokens, tgt_tokens, output_path)


def _draw_all_attention(self, src_tokens, tgt_tokens, attention_weights, output_prefix):
    # cross attention
    for layer_id in range(attention_weights.n_layers):
        for head_id in range(attention_weights.n_heads):
            output_path = output_prefix + "_layer-{}_head-{}.jpg".format(layer_id, head_id)
            draw_attention(attention_weights.get_attention(sentence_id=0, layer_id=layer_id, head_id=head_id).cpu().numpy(), src_tokens, tgt_tokens, output_path

def draw_attention(attention_matrix, source_tokens, target_tokens, output_path):
    """ save attention heatmap to a given path
    Params:
        attention_matrix: 2d numpy array, matrix[i][j] means the attention to source_tokens[j] when the input is target_tokens[i]
        source_tokens: list of source tokens
        target_tokens: list of target tokens
        output_path: path to save the heatmap
    """
    #  the output matrix is a little different from the input matrix in that the output matrix[i][j] 
    #  means the attention when output target_tokens[i] to source_tokens[j]
    #  so we do the following things to the input
    #  this is a better way to draw attention
    attention_matrix = attention_matrix[:-1,:]
    target_tokens = target_tokens[1:]

    data_frame = pandas.DataFrame(attention_matrix, index=target_tokens, columns=source_tokens)
    plt.figure(figsize=(10,10))
    seaborn.heatmap(
        data=data_frame,
        annot=False,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        linewidths=.5
    )
    plt.savefig(output_path)
    plt.close()