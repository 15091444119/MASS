import unittest
from src.combiner.splitter import encode_word, RandomBpeSplitter,  CharSplitter


class TestBpe(unittest.TestCase):
    """ Test src.combiner.bpe_helper.py """
    def setUp(self):
        self.bpe_codes = {("a", "b"): 0, ("a", "c</w>"): 1, ("ab", "ac</w>"): 2}
        self.word = "abac"

    def test_encode(self):
        ref = ["abac"]
        self.assertEqual(encode_word(self.word, self.bpe_codes), ref)

    def test_encode_count(self):
        ref = (["abac"], 3)
        self.assertEqual(encode_word(self.word, self.bpe_codes, None, True), ref)

    def test_encode_half(self):
        ref = (["ab@@", "ac"], 2)
        self.assertEqual(encode_word(self.word, self.bpe_codes, 2, True), ref)

    def test_random_bpe_word(self):
        bpe_applier = RandomBpeSplitter(self.bpe_codes)
        res = bpe_applier.split_word("abac")
        self.assertTrue(res in [["a@@", "b@@", "a@@", "c"], ["ab@@", "a@@", "c"], ["ab@@", "ac"]])

        # inference tokenize
        self.assertEqual(bpe_applier.inference_tokenize("abac"), ["abac"])
        self.assertEqual(bpe_applier.inference_tokenize("a"), ["a"])

    def test_re_encode_sentence(self):
        """ this need human check """
        bpe_applier = RandomBpeSplitter(self.bpe_codes)
        input_sentence = ["abac", "a@@", "c",  "abac", "abac"]
        res = bpe_applier.re_encode_sentence(input_sentence)
        print(input_sentence, res)

    def test_char_splitter(self):
        word_splitter = CharSplitter()
        self.assertEqual(['a@@', 'b@@', 'a@@', 'c'], word_splitter.split_word("abac"))

        # inference tokenize
        self.assertEqual(['a@@', 'b@@', 'a@@', 'c'], word_splitter.inference_tokenize("abac"))
        self.assertEqual(["a"], word_splitter.inference_tokenize("a"))
