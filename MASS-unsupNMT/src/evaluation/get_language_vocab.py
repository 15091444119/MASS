import sys
import argparse

def word_count(path):
    word_count = {}
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx % 10000 == 0:
                print(idx, file=sys.stderr)
            line = line.rstrip().split(' ')
            for word in line:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1
        return word_count

def get_language_vocab(src_path, tgt_path):
    src_word_count = word_count(src_path)
    tgt_word_count = word_count(tgt_path)

    src_word_sum = sum(src_word_count.values())
    tgt_word_sum = sum(tgt_word_count.values())

    src_vocab = set()
    tgt_vocab = set()

    for word, count in src_word_count.items():
        # don't use subword
        if "@@" in word:
            continue
        if word in tgt_word_count:
            if count / src_word_sum <= tgt_word_count[word] / tgt_word_sum:
                continue
        src_vocab.add(word)

    for word, count in tgt_word_count.items():
        if "@@" in word:
            continue
        if word in src_word_count:
            if count / tgt_word_sum < src_word_count[word] / src_word_sum:
                continue
        tgt_vocab.add(word)

    assert len(src_vocab.intersection(tgt_vocab)) == 0

    return src_vocab, tgt_vocab


def get_embedding_vocab(emb_path):
    vocab = set()
    with open(emb_path, 'r') as f:
        f.readline()
        for line in f:
            word, emb = line.rstrip().split(' ', 1)
            if "@@" in word:
                continue
            vocab.add(word)

    return vocab

def output_vocab(vocab, output_path):
    with open(output_path, 'w') as f:
        for word in vocab:
            f.writelines(word + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_text")
    parser.add_argument("--tgt_text")
    parser.add_argument("--src_emb")
    parser.add_argument("--tgt_emb")
    parser.add_argument("--src_out")
    parser.add_argument("--tgt_out")
    args = parser.parse_args()

    

    src_text_vocab, tgt_text_vocab = get_language_vocab(args.src_text, args.tgt_text)

    src_emb_vocab = get_embedding_vocab(args.src_emb)
    tgt_emb_vocab = get_embedding_vocab(args.tgt_emb)
    src_vocab = src_text_vocab.intersection(src_emb_vocab)
    tgt_vocab = tgt_text_vocab.intersection(tgt_emb_vocab)

    output_vocab(src_vocab, args.src_out)
    output_vocab(tgt_vocab, args.tgt_out)


