import json
import os

import nltk
from tqdm import tqdm

# The original implementation uses SST, POM, WikiText-2, Reddit, Meld, and
# News-200.
DATASET_NAMES = ["gender_corpus", "nationality_corpus", "religion_corpus"]


def load_sentence_debias_data(persistent_dir, bias_type):
    data = []
    for dataset_name in DATASET_NAMES:
        dataset = _GenericDataset(persistent_dir, bias_type, dataset_name)
        data.extend(dataset.load_examples())
    return data


def _gender_augment_func(sent1, sent2, examples):

    examples.append(
                {"female_example": sent1, "male_example": sent2}
            )

    return examples


def _nationality_augment_func(sent1, sent2, examples):

    examples.append(
                {
                    "nationality_example_bias": sent1,
                    "nationality_example": sent2
                }
            )

    return examples


def _religion_augment_func(sent1, sent2, examples):

    examples.append(
                {
                    "religion_example_bias": sent1,
                    "religion_example": sent2
                }
            )

    return examples


class _SentenceDebiasDataset:

    _bias_type_to_func = {
        "gender": _gender_augment_func,
        "race": _nationality_augment_func,
        "religion": _religion_augment_func,
    }

    def __init__(self, persistent_dir, bias_type):
        self._persistent_dir = persistent_dir
        self._bias_type = bias_type
        self._augment_func = self._bias_type_to_func[self._bias_type]

        self._root_data_dir = f"{self._persistent_dir}/data"

    def load_examples(self):
        raise NotImplementedError("load_examples method not implemented.")



class _GenericDataset(_SentenceDebiasDataset):
    def __init__(self, persistent_dir, bias_type, name):
        super().__init__(persistent_dir, bias_type)
        self._name = name
        self._data_file = f"{self._root_data_dir}/{name}.txt"

    def load_examples(self):
        examples = []

        with open(self._data_file, "r") as f:
            lines = f.readlines()

        data = []
        for line in tqdm(lines[:10000], desc=f"Loading data {self._name}", leave=False):
            data.extend(line)

        for i in range(0, len(data)-1, 2):
            examples = self._augment_func(data[i], data[i+1], examples)

        return examples
