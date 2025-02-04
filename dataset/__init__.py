from typing import List
from utils.instance import Instance, InstanceList

from .UIT_VSFC import UIT_VSFC_Dataset_Topic, UIT_VSFC_Dataset_Sentiment
from .UIT_ViCTSD import UIT_ViCTSD_Dataset_Toxic, UIT_ViCTSD_Dataset_Construct
from .UIT_ViOCD import UIT_ViOCD_Dataset_Domain, UIT_ViOCD_Dataset_Label
from .phoner import PhoNER
from .UIT_ViSFD import UIT_ViSFD_Dataset_ABSA


def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)