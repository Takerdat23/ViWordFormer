from builders.task_builder import build_task
from configs11.utils import get_config
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)
args = parser.parse_args()
config_file = args.config_file

if __name__ == "__main__":
    config = get_config(config_file)
    task = build_task(config)
    vocab = task.load_vocab(config.vocab)
    config.vocab.vocab_size = vocab.total_tokens
    print(config)
    # task.start()
    # task.get_predictions(task.test_dataset)
    # task.logger.info("Task done!")



