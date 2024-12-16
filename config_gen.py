import os
import yaml

# Directory for configs and shell scripts
META_DATA = {
    "name": "UIT_VFSC",
    "vocab": [
        "BPETokenizer_VSFC_Sentiment",
        "WordPieceTokenizer_VSFC_Sentiment",
        "UnigramTokenizer_VSFC_Sentiment",
        "UnigramTokenizer_VSFC_Sentiment"
    ],
    "vocab_size": 230,
    "task": {
        "sentiment": "UIT_VSFC_Dataset_Sentiment",
        "topic": "UIT_VSFC_Dataset_Topic"
    },
    "train": "data/UIT-VSFC/UIT-VSFC-train.json",
    "dev": "data/UIT-VSFC/UIT-VSFC-dev.json",
    "test": "data/UIT-VSFC/UIT-VSFC-test.json",
}

MODEL_NAMES = ["GRU", "BiGRU", "LSTM", "BiLSTM"]
TOKENIZERS = ["bpe", "unigram", "wordpiece"]
CONFIG_BASE_DIR = "scripts"
SCHEMAS = [1, 2]

CONFIG_DIR = f"{CONFIG_BASE_DIR}/{DATASET_NAME}/{MODEL_NAME}"
SHELL_SCRIPT = f"{CONFIG_DIR}/run_configs.sh"

# Ensure directories exist
os.makedirs(CONFIG_DIR, exist_ok=True)

# Base YAML configuration
def get_base_config():
    return {
        "vocab": {
            "type": "",
            "model_prefix": "",
            "model_type": "",

            "vocab_size": META_DATA["vocab_size"],
            "path": {
                "train": META_DATA["train"],
                "dev": META_DATA["dev"],
                "test": META_DATA["test"],
            },

            "eos_token": "<e>",
            "unk_token": "<u>",
            "pad_token": "<p>",
            "bos_token": "<b>",
            "space_token": "<s>",
            "min_freq": 5,
            "schema": 1,
        },
        "dataset": {
            "train": {
                "type": "",
                "path": META_DATA["train"],
            },
            "dev": {
                "type": "",
                "path": META_DATA["dev"],
            },
            "test": {
                "type": "",
                "path": META_DATA["test"],
            },
            "batch_size": 32,
            "num_workers": 16,
        },
        "model": {
            "name": "",
            "architecture": "",

            "layer_dim": 6,
            "input_dim": 256,
            "hidden_dim": 256,
            "d_model": 256,
            "dropout": 0.2,
            "output_dim": 4,
            "label_smoothing": 0.1,
            "device": "cuda",
        },
        "training": {
            "checkpoint_path": "", #"checkpoints/UIT_VFSC/Topic/BiGRU/wordpiece",
            "seed": 42,
            "learning_rate": 0.1,
            "warmup": 1000,
            "patience": 10,
            "score": "f1",
        },
        "task": "RNN_Label_Task",
    }

# Generate YAML files
def generate_yaml_files():
    yaml_files = []
    base_config = get_base_config()

    for task in META_DATA['task'].keys():
        for schema in SCHEMAS:
            for model in MODEL_NAMES:
                config_name = f"config_{task}_s{schema}__{architecture}.yaml"
                config_path = os.path.join(CONFIG_DIR, config_name)


        for t in types:
            for vocab_type in vocab_types:
                for architecture in architectures:
                    config_name = f"config_s{schema}_{t}_{vocab_type}_{architecture}.yaml"
                    config_path = os.path.join(CONFIG_DIR, config_name)

                    # Modify base config for specific schema, type, vocab, and architecture
                    base_config["vocab"]["schema"] = schema
                    base_config["vocab"]["model_type"] = t
                    base_config["vocab"]["type"] = vocab_type
                    base_config["model"]["architecture"] = architecture

                    # Update checkpoint path and model prefix
                    base_config["training"]["checkpoint_path"] = (
                        f"checkpoints/{DATASET_NAME}/Topic/{architecture}/{t}"
                    )
                    base_config["vocab"]["model_prefix"] = (
                        f"{architecture}_{t}_{vocab_type}_s{schema}"
                    )

                    # Write YAML file
                    with open(config_path, "w") as yaml_file:
                        yaml.dump(base_config, yaml_file, default_flow_style=False)

                    yaml_files.append((config_path, schema))

    return yaml_files

# Generate shell script
def generate_shell_script(yaml_files):
    with open(SHELL_SCRIPT, "w") as sh_file:
        sh_file.write("#!/bin/bash\n\n")
        for config_path, schema in yaml_files:
            sh_file.write(f"python main_s1.py --config-file {config_path} --schema {schema}\n")

# Main execution
if __name__ == "__main__":
    # Generate YAML files
    yaml_files = generate_yaml_files()

    # Generate shell script
    generate_shell_script(yaml_files)

    print(f"Generated {len(yaml_files)} YAML files and shell script: {SHELL_SCRIPT}")
