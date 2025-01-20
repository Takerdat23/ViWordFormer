import os
import yaml

# Directory for configs and shell scripts
META_DATA = {
    "name": "UIT_ViOCD",
    "task": {
        "domain": {
            "name": "UIT_ViOCD_Dataset_Domain",
            "text": "review",
            "label": "domain",
            "num_label": 4,
        },
        "topic": {
            "name": "UIT_ViOCD_Dataset_Label",
            "text": "review",
            "label": "label",
            "num_label": 2,
        }
    },
    "vocab_size": 473,
    "train": "data/UIT-ViOCD/train.json",
    "dev": "data/UIT-ViOCD/dev.json",
    "test": "data/UIT-ViOCD/test.json",
}

SCHEMAS = [1, 2]
ARCHITECTURES = ['RNNmodel']
MODEL_NAMES = ["GRU", "BiGRU", "LSTM", "BiLSTM"]
TOKENIZERS = {"bpe": "BPETokenizer", 
              "unigram": "UnigramTokenizer", 
              "wordpiece": "WordPieceTokenizer", 
              "vipher": "VipherTokenizer"
}

# Base YAML configuration
def get_base_config():
    return {
        "vocab": {
            "type": "",
            "model_prefix": "",
            "model_type": "",
            "schema": -1,
            "text": "",
            "label": "",

            "vocab_size": META_DATA["vocab_size"],
            "path": {
                "train": META_DATA["train"],
                "dev": META_DATA["dev"],
                "test": META_DATA["test"],
            },

            "unk_piece": "<unk>",
            "bos_piece": "<s>",
            "eos_piece": "</s>",
            "pad_piece": "<pad>",
            "space_token": "<space>",

            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 3,
            "space_id": 4,

            "min_freq": 5,
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
            "batch_size": 64,
            "num_workers": 6,
        },
        "model": {
            "name": "",
            "architecture": "",
            "model_type": "",
            "bidirectional": -1, #2 for bidirectional and 1 for unidirectional
            "num_output": -1,

            "num_layer": 6,
            "input_dim": 256,
            "d_model": 256,
            "dropout": 0.2,            
            "label_smoothing": 0.1,
            "device": "cuda",
        },
        "training": {
            "checkpoint_path": "", #"checkpoints/UIT_VFSC/Topic/BiGRU/wordpiece",
            "seed": 42,
            "learning_rate": 0.1,
            "warmup": 500,
            "patience": 10,
            "score": "f1",
        },
        "task": "TextClassification",
    }

# Generate YAML files
def generate_yaml_files():
    yaml_files = []
    base_config = get_base_config()

    for task, task_val in META_DATA['task'].items():
        for schema in SCHEMAS:
            for architecture in ARCHITECTURES:
                for model in MODEL_NAMES:
                    base_path = f"{META_DATA['name']}/{task}/{model}/s{schema}"
                    config_path_prefix = "configs/" + base_path
                    # Ensure directories exist
                    os.makedirs(config_path_prefix, exist_ok=True)

                    for tok, tok_val in TOKENIZERS.items():
                        if tok == "vipher" and schema == 2:
                                continue
                        config_name = f"config_{tok}_{model}_{META_DATA['name']}_{task}.yaml"
                        config_path = os.path.join(config_path_prefix, config_name)

                        cp = f"checkpoints/{META_DATA['name']}/{task}/s{schema}/{model}/{tok}"
                        # Modify base config
                        base_config["vocab"]["type"] = tok_val
                        base_config["vocab"]["model_prefix"] = cp + f"/{META_DATA['name']}_{tok}"
                        base_config["vocab"]["model_type"] = tok
                        base_config["vocab"]["text"] = task_val['text']
                        base_config["vocab"]["label"] = task_val['label']
                        base_config["vocab"]["schema"] = schema

                        if tok == "vipher":
                            base_config["model"]["architecture"] = architecture + "_ViPher"
                        else:
                            base_config["model"]["architecture"] = architecture
                        base_config["model"]["model_type"] = model[2:] if 'Bi' in model else model
                        base_config["model"]["bidirectional"] = 2 if 'Bi' in model else 1
                        # base_config["model"]["tok"] = tok
                        base_config["model"]["name"] = f"{model}_Model{base_config['model']['num_layer']}layer_{META_DATA['name']}_{tok}_{task}"
                        base_config["model"]["num_output"] = task_val['num_label']

                        base_config["dataset"]["train"]["type"] = task_val['name']
                        base_config["dataset"]["dev"]["type"] = task_val['name']
                        base_config["dataset"]["test"]["type"] = task_val['name']

                        base_config["training"]["checkpoint_path"] = cp

                        with open(config_path, "w") as yaml_file:
                            yaml.dump(base_config, yaml_file, default_flow_style=False)

                        yaml_files.append(config_path)
    
    return yaml_files

# Generate shell script
def generate_shell_script(yaml_files):
    for config_path in yaml_files:
        for task in META_DATA['task'].keys():
            for model in MODEL_NAMES:
                if f"/{task}/" in config_path and f"/{model}/" in config_path:
                    path = f"scripts/{META_DATA['name']}/{task}"
                    os.makedirs(path, exist_ok=True)
                    shell_path = path + f"/{model}scripts.sh"
                    if os.path.exists(shell_path):
                        with open(shell_path, "a") as sh_file:
                            sh_file.write(f"python main_s1.py --config-file {config_path}\n")
                    else:
                        with open(shell_path, "w") as sh_file:
                            sh_file.write("#!/bin/bash\n\n")
                            sh_file.write(f"python main_s1.py --config-file {config_path}\n")

# Main execution
if __name__ == "__main__":
    # Generate YAML files
    yaml_files = generate_yaml_files()

    # # Generate shell script
    generate_shell_script(yaml_files) 