import torch
import numpy as np
from src.data.emb_combiner_data.emb_combiner_dataset import EmbCombinerDataset
from src.data.emb_combiner_data.emb_combiner_dataset import DataLoader


class EmbCombinerCollateFn(object):

    def __init__(self, dico):
        self.dico = dico

    def __call__(self, samples):
        """
        Args:
            samples: list of tuple, each tuple contains an item from the __getitem__() function of EmbCombinerDataset
        Returns:
            tuple:
                batch_splitted_words_ids: [len, bs, dim]
                splitted_words_length: [bs]
                batch_whole_word_id: [bs]
        """
        batch_whole_word_id = []
        batch_splitted_word_ids = []

        for sample in samples:
            splitted_word_ids = sample["splitted_word_ids"]
            whole_word_id = sample["whole_word_id"]
            batch_whole_word_id.append(whole_word_id)
            batch_splitted_word_ids.append(splitted_word_ids)

        batch_splitted_word_ids, splitted_words_lengths = batch_sentences(
            sentences=batch_splitted_word_ids,
            bos_index=self.dico.eos_index,
            eos_index=self.dico.eos_index,
            pad_index=self.dico.pad_index,
            batch_first=True
        )  # use eos as bos

        batch_whole_word_id = torch.LongTensor(batch_whole_word_id)

        return batch_splitted_word_ids.cuda(), splitted_words_lengths.cuda(), batch_whole_word_id.cuda()


def build_emb_combiner_dataloader(whole_word_path, splitter, dico, batch_size, shuffle=False):
    emb_combiner_dataset = EmbCombinerDataset(
        whole_word_path=whole_word_path,
        splitter=splitter,
        dico=dico
    )

    dataloader = DataLoader(
        dataset=emb_combiner_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=EmbCombinerCollateFn(dico)
    )

    return dataloader


def batch_sentences(sentences, pad_index, bos_index, eos_index, batch_first=False):
    """
    Args:
        sentences: list of sublist, each sublist contains several int
        pad_index:
        bos_index:
        eos_index:
        batch_first:
    Returns:
        tuple:
            batch_sentences: torch.LongTensor [slen, bs]
            lengths: torch.LongTensor [bs]
    """
    lengths = torch.LongTensor([len(s) + 2 for s in sentences])
    sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(pad_index)

    sent[0] = bos_index
    for i, s in enumerate(sentences):
        if lengths[i] > 2:  # if sentence not empty
            sent[1:lengths[i] - 1, i].copy_(torch.from_numpy(s.astype(np.int64)))
        sent[lengths[i] - 1, i] = eos_index

    if batch_first:
        sent = sent.transpose(0, 1)

    return sent, lengths
