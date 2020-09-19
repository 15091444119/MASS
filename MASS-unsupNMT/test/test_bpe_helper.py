import unittest
from src.combiner.bpe_helper import encode


class TestBpe(unittest.TestCase):
    """ Test src.combiner.bpe_helper.py """
    def setUp(self):
        self.bpe_codes = {("a@@", "b@@"): 0, ("a@@", "c"): 1, ("ab@@", "ac"): 2}
        self.word = "abac"

    def test_encode(self):
        ref = tuple(["abac"])
        self.assertEqual(encode(self.word, self.bpe_codes), ref)

    def test_encode_count(self):
        ref = (("abac",), 3)
        self.assertEqual(encode(self.word, self.bpe_codes, None, True), ref)

    def test_encode_half(self):
        ref = (("ab@@", "ac"), 2)
        self.assertEqual(encode(self.word, self.bpe_codes, 2, True), ref)