from torch.utils.data import Dataset

from builders.dataset_builder import META_DATASET
from vocabs.vocab import Vocab
from utils.instance import Instance
from vocabs.utils import preprocess_sentence

import json

@META_DATASET.register()
class UIT_ViHSD(Dataset):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        path: str = config.path

        self._data = self.load_data(path)
        self._vocab = vocab

    def load_data(self, json_path: str) -> dict:
        data = json.load(open(json_path,  encoding='utf-8'))
        _data = {}
        ith = 0
        for id in data:
            item = data[id]
            sentence = item["comment"]
            tokens = preprocess_sentence(sentence)
            # ignore sentence which are longer than 1000 words
            if len(tokens) > 1000:
                continue
            _data[ith] = item
            ith +=1

        return _data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Instance:
      
        item = self._data[index]
        
        sentence = item["comment"]
        
        label = item["label"]

        encoded_sentence = self._vocab.encode_sentence(sentence)
        encoded_label = self._vocab.encode_label(label)
      
        return Instance(
            input_ids = encoded_sentence,
            label = encoded_label
        )
    
