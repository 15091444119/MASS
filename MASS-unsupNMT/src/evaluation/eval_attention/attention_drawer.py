import numpy as np
import seaborn
import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys

#from matplotlib.font_manager import _rebuild
#_rebuild()

plt.rcParams['font.sans-serif']=['simhei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
mpl.use('Agg') # plot on server

"""
methods:
    all: all layer and all heads
    all_average: all layer, averaged by heads
"""

def draw_multi_layer_multi_head_attention(src_tokens, tgt_tokens, attention_weights, method, output_dir):

    all_dir=os.path.join(output_dir, "all")
    layer_dir=os.path.join(output_dir, "layer")
    head_dir=os.path.join(output_dir, "head")
    averaged_dir=os.path.join(output_dir, "averaged")

    os.mkdir(all_dir)
    os.mkdir(layer_dir)
    os.mkdir(head_dir)
    os.mkdir(averaged_dir)

    _draw_all_attention(src_tokens, tgt_tokens, attention_weights, all_dir)
    _draw_layer_attention(src_tokens, tgt_tokens, attention_weights, layer_dir)
    _draw_head_attention(src_tokens, tgt_tokens, attention_weights, head_dir)
    _draw_averaged_attention(src_tokens, tgt_tokens, attention_weights, averaged_dir)

def _draw_layer_attention(src_tokens, tgt_tokens, attention_weights, output_dir):
    """ draw attention in each layer, heads are averaged """
    for layer_id in range(attention_weights.n_layers):
        layer_attention = attention_weights.single_layer_attention(0, layer_id).cpu().numpy()
        output_path = os.path.join(output_dir, "layer{}.jpg".format(layer_id))
        draw_attention(layer_attention, src_tokens, tgt_tokens, output_path)

def _draw_averaged_attention(src_tokens, tgt_tokens, attention_weights, output_dir):
    output_path = os.path.join(output_dir, "all_average.jpg")
    averaged_attention = attention_weights.averaged_attention(0).cpu().numpy()
    draw_attention(averaged_attention, src_tokens, tgt_tokens, output_path)

def _draw_head_attention(src_tokens, tgt_tokens, attention_weights, output_dir):
    """ draw attention of each head, layers are averaged """
    for head_id in range(attention_weights.n_heads):
        head_attention = attention_weights.single_layer_attention(0, head_id).cpu().numpy()
        output_path = os.path.join(output_dir, "head{}.jpg".format(head_id))
        draw_attention(head_attention, src_tokens, tgt_tokens, output_path)

def _draw_all_attention(src_tokens, tgt_tokens, attention_weights, output_dir):
    for layer_id in range(attention_weights.n_layers):
        for head_id in range(attention_weights.n_heads):
            output_path = os.path.join(output_dir, "layer-{}_head-{}.jpg".format(layer_id, head_id))
            draw_attention(attention_weights.get_attention(sentence_id=0, layer_id=layer_id, head_id=head_id).cpu().numpy(), src_tokens, tgt_tokens, output_path)

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
