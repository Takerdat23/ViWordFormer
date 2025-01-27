from typing import List
from utils.instance import Instance, InstanceList

from .uitvsfc import UIT_VSFC_Dataset_Topic, UIT_VSFC_Dataset_Sentiment
from .uit_absa import UIT_ABSA_Dataset
from .uit_vsmec import UIT_VSMEC
from .uit_vihsd import UIT_ViHSD
from .vihos import ViHOS_Dataset
from .newuitOCD import UIT_ViOCD_newDataset_Label
from .newuitOCDdomain import UIT_ViOCD_newDataset_Domain
from .newViNLI import NLI_Dataset
from .newuitvictsdConstruct import UIT_ViCTSD_Dataset_Construct
from .newuitvictsdToxic import UIT_ViCTSD_Dataset_Toxic
from .newuitvifdsABSA import UIT_ViFDS_Dataset_ABSA
from .newViHOS import ViHOS_newDataset
from .phoner import PhoNER
from .word_segmentation import WordSegmentationDataset
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(items: List[Instance], pad_value: int=0) -> InstanceList:
    input_ids = [torch.tensor(item.input_ids) for item in items]
    labels = []
    for item in items:
      sentiment_ids = []
      for aspect_dict in item.label:
        if 'sentiment' in aspect_dict:
            sentiment_str = aspect_dict['sentiment']
            if sentiment_str == 'Positive':
              sentiment_ids.append(0)
            elif sentiment_str == 'Negative':
              sentiment_ids.append(1)
            elif sentiment_str == 'Neutral':
              sentiment_ids.append(2)
            else:
               sentiment_ids.append(-1) # if sentiment not in these types, then -1
       
      labels.append(torch.tensor(sentiment_ids))
 
    # padding cho input
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_value)
    # padding cho label
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # Important: Pad label with -1

    return {
        "input_ids": input_ids,
        "label": labels,
        "aspect": [item.label for item in items]
    }