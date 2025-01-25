from torch.utils.data import Dataset

from builders.dataset_builder import META_DATASET
from vocabs.vocab import Vocab
from utils.instance import Instance

import json

@META_DATASET.register()
class PhoNER(Dataset):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        path: str = config.path

        _data = json.load(open(path, encoding='utf-8'))
        self._data = list(_data)
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Instance:
      
        item = self._data[index]
        
        sentence = item["words"]
        sentence = "".join(sentence)
        
        label = item["tags"]
        label = [l.split("-")[-1] for l in label]

        encoded_sentence, word_to_subword_mapping = self._vocab.encode_sequence_labeling(sentence)
        mapped_label = self._vocab.align_labels_with_subwords(label, word_to_subword_mapping)
        encoded_label = self._vocab.encode_label(mapped_label)
      
        return Instance(
            input_ids = encoded_sentence,
            label = encoded_label
        )