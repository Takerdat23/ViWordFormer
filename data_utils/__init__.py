from typing import List
from utils.instance import Instance, InstanceList
from .newuitvsfc import UIT_ViSFC_newDataset_Topic
from .uitvisfc import UIT_ViSFC_Dataset_Topic
from .newuitOCD import UIT_ViOCD_newDataset_Topic


def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)