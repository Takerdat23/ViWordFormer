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
from evaluation import F1, Precision, Recall, F1_micro, Precision_micro, Recall_micro
import pickle
import torch.optim as optim
import torch.nn.functional as F

def process_aspects(aspect_lists, vocab, pad_value=0):
    """
    Converts a list of aspect dictionaries into a tensor of aspect ids.
    """
    max_aspects = max(len(aspect_list) for aspect_list in aspect_lists)
    processed_aspects = []
    for aspect_list in aspect_lists:
      aspect_ids = []
      for aspect_dict in aspect_list:
          if "aspect" in aspect_dict:
              aspect_name = aspect_dict["aspect"]
              aspect_id = vocab.get_aspect_idx(aspect_name)
              aspect_ids.append(aspect_id)
      padded_aspect_ids = aspect_ids + [pad_value] * (max_aspects - len(aspect_ids))
      processed_aspects.append(padded_aspect_ids)
    return torch.tensor(processed_aspects)


@META_TASK.register()
class GRU_ABSA_Task(BaseTask):
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
        f1_scorer = None
        precision_scorer = None
        recall_scorer = None
        if self.config.training.average == "micro":
            f1_scorer = F1_micro()
            precision_scorer = Precision_micro()
            recall_scorer = Recall_micro()
        elif self.config.training.average == "macro":
            f1_scorer = F1()
            precision_scorer = Precision()
            recall_scorer = Recall()
        else:
            raise ValueError("Invalid average type")
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
                # items = items.to(self.device)
                # # forward pass
                # input_ids = items.input_ids
                # labels = items.label
                
                # aspect_lists = [item["label"] for item in items]
                # aspects = process_aspects(aspect_lists, self.vocab).to(self.device)
                # items = items.to(self.device) # Xóa dòng này
                input_ids = items["input_ids"].to(self.device)  # Chuyển tensor vào device
                labels = items["label"]
                aspect_lists = [item for item in labels]
                aspects = process_aspects(aspect_lists, self.vocab).to(self.device)
                
                _, loss = self.model(input_ids, labels, aspects)

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
        all_aspect_label = []
        all_aspect_pred = []
        all_sentiment_label = []
        all_sentiment_pred = []

        aspect_list = self.vocab.get_aspects_label()

        with tqdm(desc='Epoch %d - Evaluating' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for items in dataloader:
                items = items.to(self.device)
                input_ids = items.input_ids
                label = items.label  # Shape: [batch_size, num_aspects]
                
                aspect_lists = [item["label"] for item in items]
                aspects = process_aspects(aspect_lists, self.vocab).to(self.device)

                logits, _ = self.model(input_ids, label, aspects)
                output = logits.argmax(dim=-1).long()

                # Mask invalid labels (e.g., where label == 0)
                mask = (label != 0)

                # Aspect presence: 1 if sentiment != 0 (ignoring -1)
                aspect_pred = (output != 0).long()
                aspect_label = (label != 0).long()

                # Apply mask to ignore invalid labels
                aspect_pred = aspect_pred[mask]
                aspect_label = aspect_label[mask]
                output = output[mask]
                label = label[mask]

                # Store predictions and labels for aspect presence and sentiment classification
                all_aspect_pred.append(aspect_pred.cpu().numpy())
                all_aspect_label.append(aspect_label.cpu().numpy())
                all_sentiment_pred.append(output.cpu().numpy())
                all_sentiment_label.append(label.cpu().numpy())
                pbar.update()

        # Convert lists to numpy arrays for easier processing
        all_aspect_label = np.concatenate(all_aspect_label, axis=0)
        all_aspect_pred = np.concatenate(all_aspect_pred, axis=0)
        all_sentiment_label = np.concatenate(all_sentiment_label, axis=0)
        all_sentiment_pred = np.concatenate(all_sentiment_pred, axis=0)

        # Overall aspect presence scores
        aspect_score = self.compute_scores(all_aspect_pred.flatten(), all_aspect_label.flatten())
        aspect_score = {metric: float(value) for metric, value in aspect_score.items()}

        # Overall sentiment classification scores
        sentiment_score = self.compute_scores(all_sentiment_pred.flatten(), all_sentiment_label.flatten())
        sentiment_score = {metric: float(value) for metric, value in sentiment_score.items()}

        return {
            'aspect': aspect_score,
            'sentiment': sentiment_score
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
        val_scores = self.evaluate_metrics(self.dev_dataloader)
        scores.append({
            "val_scores": val_scores,
            "test_scores": test_scores
        })
        with tqdm(desc='Epoch %d - Predicting' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for items in dataloader:
                items = items.to(self.device)
                input_ids = items.input_ids
                label = items.label
                aspect_lists = [item["label"] for item in items]
                aspects = process_aspects(aspect_lists, self.vocab).to(self.device)
                logits, _ = self.model(input_ids, label, aspects)
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
            # score = scores[self.score]
            aspect_score = scores['aspect'][self.score]
            sentiment_score = scores['sentiment'][self.score]
            score = (aspect_score + sentiment_score ) / 2
        

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
        
    
