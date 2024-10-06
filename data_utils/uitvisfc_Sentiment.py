from torch.utils.data import Dataset

from builders.dataset_builder import META_DATASET
from vocabs.vocab import Vocab
from utils.instance import Instance
import pandas as pd
import json

@META_DATASET.register()
class UIT_ViSFC_Dataset_Sentiment(Dataset):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        path: str = config.path

        self._data = json.load(open(path,  encoding='utf-8'))
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Instance:
      
        item = self._data[index]
        
        sentence = item["sentence"]
        
        label = item["sentiment"]

        encoded_sentence = self._vocab.encode_sentence(sentence)
        encoded_label = self._vocab.encode_label(label)
        return Instance(
            input_ids = encoded_sentence,
            label = encoded_label
        )