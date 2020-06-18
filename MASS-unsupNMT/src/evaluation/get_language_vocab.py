import sys
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
