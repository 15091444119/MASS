"""
This codes is for evaluating encoder ability to encode words which are combined by multi bpe tokens
For example:
    对不起 is bped into 对@@不起, we easily evaluate embedding of 对@@, embedding of 不起， embedding of 抱歉，
    context embedding of 对@@, context embedding of 不起, context embedding of 抱歉， average context embedding
    of 对@@, of 不起， and of 抱歉
"""
import argparse
import torch
import numpy as np
from src.utils import bool_flag, initialize_exp
from src.evaluation.utils import load_mass_model, get_token_embedding


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reloaded", type=str, help="MASS model to reload")
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    params = parser.parse_args()

    return params

def main(params):

    initialize_exp(params)

    dico, model_params, encoder, _ = load_mass_model(params.reloaded)

    emb1 = get_token_embedding(encoder, dico, "对@@")
    emb2 = get_token_embedding(encoder, dico, "不起")

    emb3 = get_token_embedding(encoder, dico, "抱歉")

    print(torch.nn.functional.cosine_similarity(emb1, emb3, dim=-1))
    print(torch.nn.functional.cosine_similarity(emb2, emb3, dim=-1))

