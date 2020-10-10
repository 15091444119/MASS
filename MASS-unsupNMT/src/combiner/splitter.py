import random
import numpy as np
import torch
import pdb

SEPARATOR = "@@"
WORD_END = "</w>"


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


def get_sentence_combiner_mask(mappers, origin_lengths, new_lengths):
    """
    Mask the word new been separated in the origin sentence
    And the end of word in the new sentence
    This two masks will be further used in mse training for sentence level combiner but not word level combiner
    params:
        mappers: list
            List of mappers, each mapper maps original index to a span after re tokenize

        origin_lenghts:  torch.long

        new_lengths: torch.long
    returns:
        origin_mask: torch.bool, (batch_size, max_origin_length)

        new_mask: torch.bool, (batch_size, max_new_length)
    """
    batch_size = len(origin_lengths)
    max_origin_length = torch.max(origin_lengths).item()
    max_new_length = torch.max(new_lengths).item()
    origin_mask = torch.BoolTensor(batch_size, max_origin_length).fill_(False)
    new_mask = torch.BoolTensor(batch_size, max_new_length).fill_(False)
    for i in range(batch_size):
        for j in range(origin_lengths[i].item()):
            start_idx, end_idx = mappers[i][j]
            if end_idx - start_idx > 1:  # is a word new been separated
                origin_mask[i][j] = True
                new_mask[i][end_idx - 1] = True

    return origin_mask, new_mask


class WholeWordSplitter(object):

    def __init__(self):
        pass

    @classmethod
    def build_splitter(cls, params):
        if params.splitter == "BPE":
            return RandomBpeSplitter.from_code_path(params.codes_path)
        elif params.splitter == "CHAR":
            return CharSplitter()
        else:
            return NotImplementedError

    def split_word(self, word):
        raise NotImplementedError

    def re_encode_batch_words(self, batch, lengths, dico, params):
        """
        This function is used to train a combiner, we split whole word, then use the last eos to represent the word

        Params:
            batch: torch.LongTensor, size:(3, batch_size)
                A batch of whole word

            lengths: torch.LongTensor, size(batch_size)
                All length must be 3

            dico: dictionary

            params: from argprase

        Returns:
            new_batch: torch.LongTensor, size:(max_length, batch_size)

            new_lengths: torch.LongTensor, size:(batch_size)

            origin_mask: torch.boolean, size:(batch_size, 3)
                Mask to select representation of the original word

            new_mask: torch.boolean, size:(batch_size, max_length)
                Mask of eos

        Example:
            "eos, 你好, eos" will be split into eos, 你@@ 好, eos, and the representation of the last eos is used as the
            word representation
        """
        assert (lengths == 3).all()
        batch_size = len(lengths)
        new_batch, new_lengths, _, _ = self.re_encode_batch_sentences(batch, lengths, dico, params)

        origin_mask = torch.BoolTensor(batch_size, 3).fill_(False)
        origin_mask[:, 1] = True

        new_mask = torch.BoolTensor(batch_size, torch.max(new_lengths).item()).fill_(False)
        for i in range(batch_size):
            new_mask[i, new_lengths[i].item() - 1] = True

        """
        for debug encoder input
        for i in range(batch_size):
            print("Origin:", end=" ")
            for idx in batch[:, i]:
                print(dico.id2word[idx.item()], end=" ")
            print("After:", end=" ")
            for idx in new_batch[:, i]:
                print(dico.id2word[idx.item()], end=" ")
            print(origin_mask[i])
            print(new_mask[i])
            pdb.set_trace()
        """

        return new_batch, new_lengths, origin_mask, new_mask

    def re_encode_sentence(self, sentence, kept_words=None):
        """
        for each word in the sentence, if it is a whole word, we randomly encode it
        params:
            sentence: list of strings
                list of words
            kept_words: set
                set of words which should not be re encoded, if None, an empty set is used,
                always some special tokens
        Returns:
            a list of new tokens
            and a dict to map the original word to the re encoded word
                like: {1:(1, 3)} means the first token in the origin sentence[1] == newsentence[1:3]

        """
        if kept_words is None:
            kept_words = set()

        new_sentence = []
        mapper = {}
        for idx, word in enumerate(sentence):
            start_idx = len(new_sentence)

            if word in kept_words:
                new_sentence.append(word)
            elif not word.endswith(SEPARATOR) and (idx == 0 or not sentence[idx - 1].endswith(SEPARATOR)):
                # is a whole word
                re_encoded_word = self.split_word(word)
                new_sentence.extend(re_encoded_word)
            else:
                new_sentence.append(word)

            end_idx = len(new_sentence)
            mapper[idx] = (start_idx, end_idx)

        return new_sentence, mapper

    def re_encode_batch_sentences(self, batch, lengths, dico, params):
        """
        for batch and length generated from src.data.dataset,
        we re encode it to another batch which split the word
        params:
            batch: torch.LongTensor, size:(max_len, batch_size)

            lengths: torch.LongTensor

            dico: src.data.dictionary object

            params:
                mass params
        returns:
            new_batch: torch.LongTensor (new_max_length, batch_size)

            new_lengths:

            origin_mask: words been separated in new batch are masked

            new_mask: whole word end positions are masked
        """
        new_sentences = []
        mappers = []
        batch_size = len(lengths)
        kept_words = set([dico.id2word[x] for x in
                          [params.eos_index, params.bos_index, params.unk_index, params.pad_index]])

        for i in range(batch_size):
            raw_sentence = [dico.id2word[idx.item()] for idx in batch[:lengths[i], i]]
            new_sentence, mapper = self.re_encode_sentence(raw_sentence, kept_words)
            new_sentences.append(new_sentence)
            mappers.append(mapper)

        new_lengths = [len(x) for x in new_sentences]
        new_max_length = max(new_lengths)
        new_idxs = []
        for i in range(batch_size):
            new_idxs.append(np.array([dico.index(word) for word in new_sentences[i]]))

        new_batch = torch.LongTensor(new_max_length, batch_size).fill_(params.pad_index)

        for i in range(batch_size):
            new_batch[:new_lengths[i], i].copy_(torch.from_numpy(new_idxs[i]))

        new_lengths = torch.tensor(new_lengths)

        origin_mask, new_mask = get_sentence_combiner_mask(mappers, lengths, new_lengths)

        return new_batch, new_lengths, origin_mask, new_mask


class RandomBpeSplitter(WholeWordSplitter):

    def __init__(self, bpe_codes):
        super().__init__()
        self.bpe_codes = bpe_codes

    @classmethod
    def from_code_path(cls, codes_path):
        bpe_codes = read_codes(codes_path)
        return RandomBpeSplitter(bpe_codes)

    def split_word(self, word):
        """ don't fully merge bpe
        Params:
            word: string
                the word to be splitted
        Returns:
            encoded_word: list of strings
                the splited word
        """
        assert len(word) != 1
        _, merge_num = encode_word(word, self.bpe_codes, return_merge_count=True)
        assert merge_num != 0

        random_num = random.randint(0, merge_num - 1)
        encoded_word = encode_word(word, self.bpe_codes, max_merge_num=random_num)
        return encoded_word


class CharSplitter(WholeWordSplitter):
    """ split the word into characters """

    def __init__(self):
        super().__init__()

    def split_word(self, word):
        """ split the word into characters
        Params:
            word: string
                the word to be splitted
        Returns:
            encoded_word: list of strings
                the splited word
        """
        assert len(word) != 1

        encoded_word = encode_word(word, {}, max_merge_num=0)
        return encoded_word


if __name__ == "__main__":
    bpe_codes = {("a", "b"): 0, ("a", "c</w>"): 1, ("ab", "ac</w>"): 2}
    print(encode_word("aefabghabacefatc", bpe_codes))

