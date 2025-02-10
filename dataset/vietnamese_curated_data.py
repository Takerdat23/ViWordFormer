from torch.utils.data import Dataset

from builders.dataset_builder import META_DATASET
from vocabs.vocab import Vocab
from utils.instance import Instance

import json

@META_DATASET.register()
class VTSNLP_Dataset_Pretrain(Dataset):
    def __init__(self, config, vocab: Vocab):
        super().__init__()

        path: str = config.path

        if config.get('max_len', None) is not None:
            self._max_len = config.max_len
        else:
            self._max_len = None
            
        self._data = json.load(open(path,  encoding='utf-8'))
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Instance:
      
        item = self._data[index]
        
        sentence = item["text"]
        
        encoded_sentence = self._vocab.encode_sentence(sentence, self._max_len)

      
        return Instance(
            input_ids = encoded_sentence,
        
        )
  