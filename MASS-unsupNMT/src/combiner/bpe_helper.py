import random
SEPARATOR="@@"
WORD_END="</w>"


def encode_word(orig, bpe_codes, max_merge_num=None, return_merge_count=False):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    Params:
        orig: string
            the word to be encoded
        bpe_codes: dict, like {("1", "2</w>"): 1}
        max_merge_num: int
            max number of merge operation, the function is stopped after this number
        return_merge_count: bool
            if we want to return the number of merge operation during encoding
    Returns:
        a list of tokens if return_merge_count is false
        a list of tokens and number of merge count if return_merge_count is True
            merge the same pairs at one time counts one
    """
    if len(orig) == 1:
        if return_merge_count:
            return orig, 0
        else:
            return orig

    word = list(orig[:-1]) + [orig[-1] + WORD_END]  # wordend is add to the last character

    merge_count = 0
    while len(word) > 1:

        if max_merge_num is not None and merge_count >= max_merge_num:
            break

        # get list of symbol pairs; optionally apply dropout
        pairs = [(bpe_codes[pair], i, pair) for (i, pair) in enumerate(zip(word, word[1:])) if pair in bpe_codes]

        if not pairs:
            break

        merge_count += 1

        #get first merge operation in list of BPE codes
        bigram = min(pairs)[2]

        # find start position of all pairs that we want to merge
        positions = [i for (rank,i,pair) in pairs if pair == bigram]

        i = 0
        new_word = []

        bigram = ''.join(bigram)

        for j in positions:
            # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
            if j < i:
                continue
            new_word.extend(word[i:j]) # all symbols before merged pair
            new_word.append(bigram) # merged pair
            i = j+2 # continue after merged pair
        new_word.extend(word[i:]) # add all symbols until end of word
        word = new_word


    # add separator
    output = []
    for token in word[:-1]:
        output.append(token + SEPARATOR)

    # remove word end
    assert word[-1].endswith(WORD_END)
    output.append(word[-1][:-len(WORD_END)])

    return output if not return_merge_count else (output, merge_count)


def read_codes(codes):
    bpe_codes = {}
    with open(codes, 'r') as f:
        for line in f:
            src, tgt, count = line.rstrip().split()
            assert (src, tgt) not in bpe_codes
            bpe_codes[(src, tgt)] = len(bpe_codes)
    return bpe_codes


class RandomBpeApplier(object):

    def __init__(self, bpe_codes):
        self.bpe_codes = bpe_codes

    @classmethod
    def from_code_path(cls, codes_path):
        bpe_codes = read_codes(codes_path)
        return RandomBpeApplier(bpe_codes)

    def random_encode_word(self, word):
        """ don't fully merge bpe"""
        _, merge_num = encode_word(word, self.bpe_codes, return_merge_count=True)
        assert merge_num != 0

        random_num = random.randint(0, merge_num - 1)
        print(random_num)
        encoded_word = encode_word(word, self.bpe_codes, max_merge_num=random_num)
        return encoded_word

    def re_encode_sentence(self, sentence):
        """
        for each word in the sentence, if it is a whole word, we randomly encode it
        params:
            sentence: list of strings
                list of words
        Returns:
            a list of new tokens
            and a dict to map the original word to the re encoded word
                like: {1:(1, 3)} means the first token in the origin sentence[1] == newsentence[1:3]
        """
        new_sentence = []
        mapper = {}
        for idx, word in enumerate(sentence):
            start_idx = len(new_sentence)

            if not word.endswith(SEPARATOR) and (idx == 0 or not sentence[idx - 1].endswith(SEPARATOR)):
                # is a whole word
                re_encoded_word = self.random_encode_word(word)
                new_sentence.extend(re_encoded_word)
            else:
                new_sentence.append(word)

            end_idx = len(new_sentence)
            mapper[idx] = (start_idx, end_idx)

        return new_sentence, mapper

