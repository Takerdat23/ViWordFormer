from argparse import ArgumentParser
import copy
import os
import json
from builders.task_builder import build_task
from configs.utils import get_config
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

def train_with_lr(config, learning_rate):
    config_copy = copy.deepcopy(config)
    config_copy.training.learning_rate = learning_rate
    config_copy.training.checkpoint_path = config_copy.training.checkpoint_path + f"_{learning_rate}"
    cp_path = os.path.join(config_copy.training.checkpoint_path, config_copy.model.name)
    task = build_task(config_copy)
    vocab = task.load_vocab(config_copy.vocab)
    task.start()

    task.get_predictions(task.test_dataset)
    with open(os.path.join(cp_path, "scores.json"), 'r') as f:
        scores = json.load(f)

    return scores

def tune_learning_rate_v2(config_file, min_lr=1e-5, max_lr=1e-1, k=10):
    """
    Find optimal learning rate by splitting range into k equal parts
    Args:
        config_file: path to config file
        min_lr: minimum learning rate to try
        max_lr: maximum learning rate to try 
        k: number of splits
    """
    config = get_config(config_file)
    
    learning_rates = np.logspace(np.log10(min_lr), np.log10(max_lr), k)
    results = {}
    
    best_lr = None
    best_score = 0.0

    for lr in learning_rates:
        lr = float(lr)
        print(f"\nTrying learning rate: {lr:.6f}")
        scores = train_with_lr(config, lr)[0]
        results[lr] = scores

        if scores['test_scores']["f1"] > best_score:
            best_score = scores['test_scores']["f1"]
            best_lr = lr

    # Plot results
    plt.figure(figsize=(10, 6))
    lrs = list(results.keys())
    scores = [metrics['test_scores']["f1"] for metrics in results.values()]
    title = config.model.name + " " + config.vocab.type
    plt.semilogx(lrs, scores, 'bo-')
    plt.grid(True)
    plt.xlabel('learning rate')
    plt.ylabel('score')
    plt.title(title)
    
    # Add points
    plt.scatter(lrs, scores, color='blue')
    
    # Highlight best point
    plt.scatter(best_lr, best_score, color='red', s=100, 
                label=f'Best LR: {best_lr:.6f}')
    plt.legend()

    # Save plot
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/{title}.png')
    plt.close()

    # Save results to JSON
    if not os.path.exists('results'):
        os.makedirs('results')
    with open(f'results/{title}.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    
    print("\nLearning Rate Tuning Results:")
    for lr, scores in results.items():
        print(f"LR: {lr:.6f}, Test Score: {scores['test_scores']}")
    
    print(f"\nBest learning rate found: {best_lr:.6f}")
    
    return best_lr


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--max-lr", type=float, default=1e-1)
    parser.add_argument("--num-splits", type=int, default=10)
    
    args = parser.parse_args()
    tune_learning_rate_v2(args.config_file, args.min_lr, args.max_lr, args.num_splits)