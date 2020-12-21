import torch
import pdb
from ..emb_combiner_data.emb_combiner_dataloader import batch_sentences
from src.model.combiner.context_combiner.combine_utils import get_splitted_words_mask, get_new_splitted_combine_labels
from .dataset import ContextCombinerTrainDataset, ContextCombinerTestDataset
from src.data.emb_combiner_data.emb_combiner_dataset import DataLoader


class ContextCombinerCollateFn(object):

    def __init__(self, dico):
        self.dico = dico

    def __call__(self, samples):
        """
        Args:
            samples: list of tuple, each tuple contains an item from the __getitem__() function of ContextCombinerDataset
        Returns:
            dict:
                original_batch:
                original_length:
                splitted_batch:
                splitted_length:
                trained_word_mask:
                combine_labels:
        """
        batch_original_sentence = []
        batch_splitted_sentence = []
        mappers = []

        for sample in samples:
            batch_original_sentence.append(sample["original_sentence"])
            batch_splitted_sentence.append(sample["splitted_sentence"])
            mappers.append(sample["mapper"])

        batch_original_sentence, original_length = batch_sentences(
            sentences=batch_original_sentence,
            bos_index=self.dico.eos_index,
            eos_index=self.dico.eos_index,
            pad_index=self.dico.pad_index,
            batch_first=False
        )  # use eos as bos

        batch_splitted_sentence, splitted_length = batch_sentences(
            sentences=batch_splitted_sentence,
            bos_index=self.dico.eos_index,
            eos_index=self.dico.eos_index,
            pad_index=self.dico.pad_index,
            batch_first=False
        )

        trained_word_mask = get_splitted_words_mask(mappers, original_length)

        combine_lables = get_new_splitted_combine_labels(mappers, original_length, splitted_length)


        batch = {
            "original_batch": batch_original_sentence,
            "original_length": original_length,
            "splitted_batch": batch_splitted_sentence,
            "splitted_length": splitted_length,
            "trained_word_mask": trained_word_mask,
            "combine_labels": combine_lables
        }

        return batch


def build_context_combiner_train_data_loader(data_path, dico, splitter, batch_size):
    data_set = ContextCombinerTrainDataset(
        labeled_data_path=data_path,
        dico=dico,
        splitter=splitter
    )

    dataloader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ContextCombinerCollateFn(dico)
    )

    return dataloader


def build_context_combiner_test_data_loader(data_path, dico, splitter, batch_size):
    data_set = ContextCombinerTestDataset(
        labeled_data_path=data_path,
        dico=dico,
        splitter=splitter
    )

    dataloader = DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ContextCombinerCollateFn(dico)
    )

    return dataloader


def load_data(params, dico, splitter):
    train_data = build_context_combiner_train_data_loader(
        data_path=params.combiner_train_data,
        dico=dico,
        splitter=splitter,
        batch_size=params.batch_size
    )

    dev_data = build_context_combiner_test_data_loader(
        data_path=params.combiner_dev_data,
        dico=dico,
        splitter=splitter,
        batch_size=params.batch_size
    )

    test_data = build_context_combiner_test_data_loader(
        data_path=params.combiner_test_data,
        dico=dico,
        splitter=splitter,
        batch_size=params.batch_size
    )

    data = {
        "train": train_data,
        "dev": dev_data,
        "test": test_data
    }

    return data
