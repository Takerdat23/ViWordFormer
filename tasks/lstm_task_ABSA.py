from torch import Tensor
from torch.utils.data import DataLoader
from builders.model_builder import  build_model
from builders.vocab_builder import build_vocab
from torch.optim.lr_scheduler import LambdaLR
import os
import torch
from shutil import copyfile
import numpy as np
from tqdm import tqdm
import json
import math
from utils.logging_utils import setup_logger
from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
from tasks.base_task import BaseTask
from data_utils import collate_fn
from evaluation import F1, Precision, Recall
import pickle
@META_TASK.register()
class lstm_ABSA_Task(BaseTask):
    def __init__(self, config):
        super().__init__(config)

    def configuring_hyperparameters(self, config):
        self.epoch = 0
        self.score = config.training.score
        self.learning_rate = config.training.learning_rate
        self.patience = config.training.patience
        self.warmup = config.training.warmup

    def load_datasets(self, config):
        self.train_dataset = build_dataset(config.train, self.vocab)
        self.dev_dataset = build_dataset(config.dev, self.vocab)
        self.test_dataset = build_dataset(config.test, self.vocab)

    def create_dataloaders(self, config):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )
        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=collate_fn
        )

    def create_metrics(self):
        f1_scorer = F1()
        precision_scorer = Precision()
        recall_scorer = Recall()
        self.scorers = {
            str(f1_scorer): f1_scorer,
            str(precision_scorer): precision_scorer,
            str(recall_scorer): recall_scorer
        }

    def compute_scores(self, inputs: Tensor, labels: Tensor) -> dict:
        scores = {}
        for scorer_name in self.scorers:
            scores[scorer_name] = self.scorers[scorer_name].compute(inputs, labels)

        return scores
    
   
    
    def get_vocab(self): 
        return self.vocab


    def train(self):
        self.model.train()

        running_loss = .0
        with tqdm(desc='Epoch %d - Training' % self.epoch, unit='it', total=len(self.train_dataloader)) as pbar:
            for it, items in enumerate(self.train_dataloader):
                items = items.to(self.device)
                # forward pass
                input_ids = items.input_ids
        
                labels = items.label
                _, loss = self.model(input_ids, labels)
                
                # backward pass
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                running_loss += loss.item()

                # update the training status
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                self.scheduler.step()

    def evaluate_metrics(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        all_labels = []
        all_predictions = []
      
        aspect_list = self.vocab.get_aspects_label()
        aspect_wise_scores = {f"{i}": {} for i in aspect_list}  # To store scores (F1, recall, precision) for each aspect
        total_f1, total_recall, total_precision = 0, 0, 0   # To store accuracy for each aspect

        with tqdm(desc='Epoch %d - Evaluating' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for items in dataloader:
                items = items.to(self.device)
                input_ids = items.input_ids
                label = items.label  # Shape: [batch_size, num_aspects]
                
                logits, _ = self.model(input_ids, label)  
                output = logits.argmax(dim=-1).long() 
                
                all_labels.append(label.cpu().numpy())  # Store the true labels
                all_predictions.append(output.cpu().numpy())  # Store the predictions
                
                pbar.update()
                
       
        # Convert lists to numpy arrays for easier indexing and computation
        all_labels = np.concatenate(all_labels, axis=0)  
        all_predictions = np.concatenate(all_predictions, axis=0)  
        # Calculate accuracy for each aspect
        for i, aspect in enumerate(aspect_list):
            correct_preds = np.sum(all_predictions[:, i] == all_labels[:, i])  # Correct predictions for aspect i
            total_samples = all_labels.shape[0]  # Total number of samples
            
            preds_aspect = all_predictions[:, i]
            labels_aspect = all_labels[:, i]
            
            # Compute additional metrics using the compute_scores function
            aspect_scores = self.compute_scores(preds_aspect, labels_aspect)
            
            aspect_scores = {metric: float(value) if isinstance(value, np.float64) else value for metric, value in aspect_scores.items()}
            aspect_wise_scores[aspect] = aspect_scores
        
            # Accumulate scores to compute averages later
            total_f1 += aspect_scores['f1']
            total_recall += aspect_scores['recall']
            total_precision += aspect_scores['precision']
            
        num_aspects = len(aspect_list)
        avg_f1 = total_f1 / num_aspects
        avg_recall = total_recall / num_aspects
        avg_precision = total_precision / num_aspects

        return {
        'aspect_wise_scores': aspect_wise_scores,
        'f1': avg_f1,
        'recall': avg_recall,
        'precision': avg_precision
    }


    def get_predictions(self, dataset):
        if not os.path.isfile(os.path.join(self.checkpoint_path, 'best_model.pth')):
            self.logger.error("Prediction require the model must be trained. There is no weights to load for model prediction!")
            raise FileNotFoundError("Make sure your checkpoint path is correct or the best_model.pth is available in your checkpoint path")

        self.load_checkpoint(os.path.join(self.checkpoint_path, "best_model.pth"))

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            collate_fn=collate_fn
        )

        self.model.eval()
        scores = []
        labels = []
        predictions = []
        results = []
        test_scores = self.evaluate_metrics(self.test_dataloader)
        val_scores = self.evaluate_metrics(self.test_dataloader)
        scores.append({
            "val_scores": val_scores , 
            "test_scores": test_scores
        })
        with tqdm(desc='Epoch %d - Predicting' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for items in dataloader:
                items = items.to(self.device)
                input_ids = items.input_ids
                label = items.label
                logits, _ = self.model(input_ids, label)
                output = logits.argmax(dim=-1).long()
                
                labels.append(label.cpu().numpy())
                predictions.append(output.cpu().numpy())

                sentence = self.vocab.decode_sentence(input_ids)
                label = self.vocab.decode_label(label)[0]
                prediction = self.vocab.decode_label(output)[0]

                results.append({
                    "sentence": sentence,
                    "label": label,
                    "prediction": prediction
                })
                
                
                pbar.update()
           

        self.logger.info("Test scores %s", scores)
        json.dump(scores, open(os.path.join(self.checkpoint_path, "scores.json"), "w+", encoding="utf-8"), ensure_ascii=False, indent=4)
        json.dump(results, open(os.path.join(self.checkpoint_path, "predictions.json"), "w+", encoding="utf-8"), ensure_ascii=False, indent=4)

    def start(self):
        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            best_score = checkpoint["best_score"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"] + 1
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            best_score = .0
            patience = 0

        while True:
            self.train()
            # val scores
            scores = self.evaluate_metrics(self.dev_dataloader)
            self.logger.info("Validation scores %s", scores)
            score = scores[self.score]

            # Prepare for next epoch
            is_the_best_model = False
            if score > best_score:
                best_score = score
                patience = 0
                is_the_best_model = True
            else:
                patience += 1

            # switch_to_rl = False
            exit_train = False

            if patience == self.patience:
                self.logger.info('patience reached.')
                exit_train = True

            self.save_checkpoint({
                "epoch": self.epoch,
                "best_score": best_score,
                "patience": patience,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optim.state_dict(),
                "scheduler": self.scheduler.state_dict()
            })

            if is_the_best_model:
                copyfile(
                    os.path.join(self.checkpoint_path, "last_model.pth"), 
                    os.path.join(self.checkpoint_path, "best_model.pth")
                )

            if exit_train:
                break

            self.epoch += 1
        
    
