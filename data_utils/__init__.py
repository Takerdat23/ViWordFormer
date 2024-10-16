from typing import List
from utils.instance import Instance, InstanceList

from .uitvisfc_Sentiment import UIT_ViSFC_Dataset_Sentiment
from .uitvisfc import UIT_ViSFC_Dataset_Topic
from .newuitOCD import UIT_ViOCD_newDataset_Label
from .newuitOCDdomain import UIT_ViOCD_newDataset_Domain
from .newViNLI import NLI_Dataset
from .newuitvictsdConstruct import UIT_ViCTSD_Dataset_Construct
from .newuitvictsdTopic import UIT_ViCTSD_Dataset_Toxic
from .newuitvisfdABSA import UIT_ViSFD_Dataset_ABSA



def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)

