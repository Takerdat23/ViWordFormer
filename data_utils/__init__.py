from typing import List
from utils.instance import Instance, InstanceList
from .newuitvsfc import UIT_ViSFC_newDataset_Topic
from .newVSFC_Sentiment import UIT_ViSFC_newDataset_Sentiment
from .uitvisfc import UIT_ViSFC_Dataset_Topic
from .newuitOCD import UIT_ViOCD_newDataset_Label
from .newuitOCDdomain import UIT_ViOCD_newDataset_Domain
from .newViNLI import ViNLI_newDataset_Topic


def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)