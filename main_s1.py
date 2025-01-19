from builders.task_builder import build_task
from configs.utils import get_config
from argparse import ArgumentParser


import random, os, torch
import numpy as np
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

parser = ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)
# parser.add_argument("--schema", type=int, required=True)
args = parser.parse_args()
config_file = args.config_file

if __name__ == "__main__":
    config = get_config(config_file)
    # config.vocab.schema = args.schema
    seed_everything(config.training.seed)
    task = build_task(config)
    task.start()
    task.get_predictions(task.test_dataset)
    task.logger.info("Task done!")
