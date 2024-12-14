from typing import List
from utils.instance import Instance, InstanceList

from .UIT_VSFC import UIT_VSFC_Dataset_Topic, UIT_VSFC_Dataset_Sentiment

def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)