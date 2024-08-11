from typing import List
from utils.instance import Instance, InstanceList

from .uitvisfc import UIT_ViSFC_Dataset_Topic

def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)