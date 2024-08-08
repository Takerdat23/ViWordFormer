from .registry import Registry

META_DATASET = Registry("DATASET")

def build_dataset(config, vocab):
    dataset = META_DATASET.get(config.type)(config, vocab)

    return dataset