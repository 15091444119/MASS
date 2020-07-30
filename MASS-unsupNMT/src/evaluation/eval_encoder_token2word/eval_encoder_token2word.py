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
from src.evaluation.utils import load_mass_model, get_token_embedding, encode_word 


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reloaded", type=str, help="MASS model to reload")
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    params = parser.parse_args()

    return params

def get_embedding_similarity_online(encoder, dico):
    while(True):
        print("Input two words")
        try:
            word1, word2 = input().rstrip().split()
        except:
            continue
        if word1 not in dico.word2id:
            print("{} not in dictionary".format(word1))
            continue
        if word2 not in dico.word2id:
            print("{} not in dictionary".format(word2))
            continue
        emb1 = get_token_embedding(encoder, dico, word1)
        emb2 = get_token_embedding(encoder, dico, word2)
        print(torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1))

def context_representation_similarity_online(encoder, dico, mass_params):
    while(True):

        print("Input tokens")
        tokens = input().rstrip().split()
        for token in tokens:
            assert token in dico.word2id, token
        print("Input a word")
        word = input().rstrip()
        assert word in dico.word2id, word

        encoded_tokens = encode_word(encoder, dico, mass_params, tokens, mass_params.lang2id["zh"])
        encoded_word = encode_word(encoder, dico, mass_params, [word], mass_params.lang2id["zh"])
        word_representation = encoded_word[1] # index 0 is the eos        
        average_smiliarity = 0

        for token, token_representation in zip(tokens, encoded_tokens[1:-1]):  # index 0 and -1 are eos 
            similarity = torch.nn.functional.cosine_similarity(token_representation, word_representation, dim=-1)
            average_smiliarity += similarity
            print("{} {} -> {}".format(token, word, similarity))

        average_smiliarity = average_smiliarity / len(tokens)
        print("Average similarity: {}".format(average_smiliarity))
        

def main(params):

    initialize_exp(params)

    dico, model_params, encoder, _ = load_mass_model(params.reloaded)

    get_embedding_similarity_online(encoder, dico)


if __name__ == "__main__":
    params = parse_params()
    main(params)
