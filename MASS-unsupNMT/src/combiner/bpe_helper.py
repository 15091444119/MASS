def encode(orig, bpe_codes, max_merge_num=None, return_merge_count=False):
    """Encode word based on list of BPE merge operations(fasttext style), which are applied consecutively
    Params:
        orig: string
            the word to be encoded
        bpe_codes: dict, like {("1@@", "2"): 1}
        max_merge_num: int
            max number of merge operation
        return_merge_count: bool
            if we want to return the number of merge operation during encoding
    Returns:
        a tuple of the encoded word if return_merge_count is false
        a tuple of the encoded word and number of merge count if return_merge_count is True
            merge the same pairs at one time counts one
    """
    if len(orig) == 1:
        if return_merge_count:
            return orig, 0
        else:
            return orig

    word = [char + "@@" for char in orig[:-1]] + [orig[-1]]

    merge_count = 0
    while len(word) > 1:

        # get list of symbol pairs; optionally apply dropout
        pairs = [(bpe_codes[pair],i,pair) for (i,pair) in enumerate(zip(word, word[1:])) if pair in bpe_codes]

        if not pairs:
            break

        #get first merge operation in list of BPE codes
        bigram = min(pairs)[2]

        # find start position of all pairs that we want to merge
        positions = [i for (rank,i,pair) in pairs if pair == bigram]

        i = 0
        new_word = []

        if bigram[-1].endswith("@@"):
            bigram = ''.join(bigram).replace("@@", "") + "@@"
        else:
            bigram = ''.join(bigram).replace("@@", "")

        for j in positions:
            # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
            if j < i:
                continue
            new_word.extend(word[i:j]) # all symbols before merged pair
            new_word.append(bigram) # merged pair
            i = j+2 # continue after merged pair
        new_word.extend(word[i:]) # add all symbols until end of word
        word = new_word

        # increas merge count and check if continuing to merge
        merge_count += 1
        if max_merge_num is not None and merge_count >= max_merge_num:
            break

    word = tuple(word)

    return word if not return_merge_count else (word, merge_count)

