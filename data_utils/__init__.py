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


def collate_fn(items: List[Instance], pad_value: int=0) -> InstanceList:
    return InstanceList(items, pad_value)
