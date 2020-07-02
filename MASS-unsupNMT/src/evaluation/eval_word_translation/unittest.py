import unittest
from .alignment import extract_word_types, WHOLEWORD, SEPERATEDWORD, calculate_whole_word_seperated_word_translation_acc, filter_alignment_one2one

class TestAlignment(unittest.TestCase):

    def test_extract_word_types(self):
        input_case = ["1@@ 2 3 4@@ 5 6", "1@@ 2@@ 3"]
        self.assertEqual(extract_word_types(input_case), [[SEPERATEDWORD, WHOLEWORD, SEPERATEDWORD], [SEPERATEDWORD]])

    def test_calculate_whole_word_seperated_word_translation_acc(self):
        alignments = ["0-0 0-1 1-2 2-3 3-4 4-5"]
        srcs = ["这是 一个 测试 函数 样例 ."] # srcbpe=["这是 一个 测@@ 试 函@@ 数 样@@ 例 ."]
        tgts = ["here is a test function case ."]
        hyps= ["here was two exam program case ."]
        word_types=[[WHOLEWORD, WHOLEWORD, SEPERATEDWORD, WHOLEWORD]]

        # 这是，因为是一对多，所以不算，只有1-2， 2-3， 3-4 进行了计算
        # 其中测试是一个被bpe切分的，所以一共有2个whole(一个，句号)，3个seperate（测试, 函数， 题目, ）
        # whole 的 acc是0.5（句号做对了）seperate的acc是0.33 (case对了) 

        whole_word_acc, seperated_word_acc =  calculate_whole_word_seperated_word_translation_acc(alignments, srcs, tgts, hyps, word_types)
        self.assertEqual(whole_word_acc, 0.5)
        self.assertEqual(seperated_word_acc, 1 / 3)

    def test_filter_alignment_one2one(self):
        alignment = "0-0 0-1 1-2 2-2 3-2 4-4 5-5"
        
        self.assertEqual(filter_alignment_one2one(alignment), "4-4 5-5")
