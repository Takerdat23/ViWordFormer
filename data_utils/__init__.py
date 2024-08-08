from typing import List
from utils.instance import Instance, InstanceList

def collate_fn(items: List[Instance]) -> InstanceList:
    return InstanceList(items)