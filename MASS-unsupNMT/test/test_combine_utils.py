import unittest
import torch
from src.combiner.constant import PAD, COMBINE_END, COMBINE_FRONT, NOT_COMBINE
from src.combiner.combine_utils import get_combine_labels, get_length_after_combine, get_masks_for_combine_reps, gather_splitted_combine_representation, get_mask_for_select_combined_rep


class test_dico(object):

    def __init__(self):
        self.pad_index = 1
        self.id2word = {
            0: "<EOS>",
            1: "<PAD>",
            2: "hel@@",
            3: "lo",
            4: "wor@@",
            5: "ld",
            6: ".",
            7: "haha",
            8: "<MASK>"
        }
        self.word2id = {word: index for index, word in self.id2word.items()}


class Test(unittest.TestCase):

    def setUp(self):
        self.dico = test_dico()
        self.sentences = [
            ["<EOS>", "hel@@", "lo", "<MASK>", "<MASK>", "wor@@", "ld", ".", "<EOS>", "<PAD>", "<PAD>"],
            ["<EOS>", "<MASK>", "<MASK>", "<MASK>", "<MASK>", "<MASK>", "lo", "wor@@", "ld", ".", "<EOS>"]
        ]
        self.batch = torch.tensor([
            [self.dico.word2id[word] for word in sentence] for sentence in self.sentences
        ]).long().transpose(0, 1)

        combine_labels = [
            [NOT_COMBINE, COMBINE_FRONT, COMBINE_END, NOT_COMBINE, NOT_COMBINE, COMBINE_FRONT, COMBINE_END, NOT_COMBINE, NOT_COMBINE, PAD, PAD],
            [NOT_COMBINE, NOT_COMBINE, NOT_COMBINE, NOT_COMBINE, NOT_COMBINE, NOT_COMBINE, NOT_COMBINE, COMBINE_FRONT, COMBINE_END, NOT_COMBINE, NOT_COMBINE]
        ]
        self.combine_labels = torch.tensor(combine_labels).long()
        self.final_length = torch.tensor([7, 10]).long()
        self.final_rep_using_combined_rep_mask = torch.tensor([
            [False, True, False, False, True, False, False, False, False, False],
            [False, False, False, False, False, False, False, True, False, False]
        ]).bool()

        self.final_rep_using_splitted_rep_mask = torch.tensor([
            [True, False, True, True, False, True, True, False, False, False],
            [True, True, True, True, True, True, True, False, True, True]
        ]).bool()

        self.splitted_rep_for_final_rep_mask = torch.tensor([
            [True, False, False, True, True, False, False, True, True, False, False],
            [True, True, True, True, True, True, True, False, False, True, True]
        ]).bool()

        # representation with one dimension for testing
        self.splitted_rep = self.splitted_rep_for_final_rep_mask.long().float().unsqueeze(-1)
        self.combined_rep = torch.tensor([2, 2, 2]).unsqueeze(-1).float()
        # 1 means from splitted, 2 means from combined, 0 means pad
        self.final_rep = torch.tensor([
            [1, 2, 1, 1, 2, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 2, 1, 1]
        ]).unsqueeze(-1).float()

        self.new_generated_subwords_labels = torch.tensor([
            [NOT_COMBINE, COMBINE_FRONT, COMBINE_END, NOT_COMBINE, NOT_COMBINE, NOT_COMBINE, NOT_COMBINE, NOT_COMBINE,
             NOT_COMBINE, PAD, PAD],
            [NOT_COMBINE, NOT_COMBINE, NOT_COMBINE, NOT_COMBINE, NOT_COMBINE, NOT_COMBINE, NOT_COMBINE, COMBINE_FRONT,
             COMBINE_END, NOT_COMBINE, NOT_COMBINE]
        ])

        self.selecting_mask = torch.tensor([0, 2])

    def test_get_length_after_combine(self):
        hyp = get_length_after_combine(self.combine_labels)

        self.assertTrue(torch.eq(hyp, self.final_length).all())

    def test_get_combine_labels(self):
        hyp = get_combine_labels(self.batch, self.dico)

        self.assertTrue(torch.eq(self.combine_labels, hyp).all())

    def test_get_mask_for_decoder(self):
        hyp = get_length_after_combine(self.combine_labels)

        self.assertTrue(torch.eq(self.final_length, hyp).all())

    def test_get_masks_for_combine_reps(self):
        hyp_final_rep_using_splitted_rep_mask, hyp_final_rep_using_combined_rep_mask, hyp_splitted_rep_for_final_rep_mask = get_masks_for_combine_reps(
            combine_labels=self.combine_labels,
            final_length=self.final_length
        )
        self.assertTrue(torch.eq(hyp_final_rep_using_combined_rep_mask, self.final_rep_using_combined_rep_mask).all())
        self.assertTrue(torch.eq(hyp_final_rep_using_splitted_rep_mask, self.final_rep_using_splitted_rep_mask).all())
        self.assertTrue(torch.eq(hyp_splitted_rep_for_final_rep_mask, self.splitted_rep_for_final_rep_mask).all())

    def test_gather(self):
        final_rep = gather_splitted_combine_representation(
            splitted_rep=self.splitted_rep,
            combined_rep=self.combined_rep,
            final_length=self.final_length,
            final_rep_using_splitted_rep_mask=self.final_rep_using_splitted_rep_mask,
            final_rep_using_combined_rep_mask=self.final_rep_using_combined_rep_mask,
            splitted_rep_for_final_rep_mask=self.splitted_rep_for_final_rep_mask
        )
        self.assertTrue(torch.eq(final_rep, self.final_rep).all())

    def test_select_from_combined(self):

        mask = get_mask_for_select_combined_rep(self.combine_labels, self.new_generated_subwords_labels)

        self.assertTrue(torch.eq(self.selecting_mask, mask).all())
