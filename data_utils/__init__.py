from typing import List
from utils.instance import Instance, InstanceList

from .uitvsfc import UIT_VSFC_Dataset_Topic, UIT_VSFC_Dataset_Sentiment
from .uit_absa import UIT_ABSA_Dataset
from .newuitOCD import UIT_ViOCD_newDataset_Label
from .newuitOCDdomain import UIT_ViOCD_newDataset_Domain
from .newViNLI import NLI_Dataset
from .newuitvictsdConstruct import UIT_ViCTSD_Dataset_Construct
from .newuitvictsdToxic import UIT_ViCTSD_Dataset_Toxic
from .newuitvifdsABSA import UIT_ViFDS_Dataset_ABSA
from .newViHOS import ViHOS_newDataset



def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)

