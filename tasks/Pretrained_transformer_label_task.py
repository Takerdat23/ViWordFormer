from torch import Tensor
from torch.utils.data import DataLoader
import os
from typing import List, Dict, Any
from shutil import copyfile
import numpy as np
from tqdm import tqdm
import json
from transformers import Trainer, TrainingArguments, EvalPrediction
from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
from tasks.base_task import BaseTask
from utils.instance import Instance, InstanceList
from evaluation import F1, Precision, Recall

def collate_fn(items: List[Dict[str, Any]], pad_value: int = 0) -> Dict[str, Tensor]:
    instances = [Instance(**item) for item in items]
    instance_list = InstanceList(instances, pad_value)
    return {
        "input_ids": instance_list.input_ids,
        "labels": instance_list.label
    }


@META_TASK.register()
class Pretrained_TransformerLabel(BaseTask):
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
    def compute_scores(self, eval_pred: EvalPrediction) -> dict:
      # Extract predictions and labels from eval_pred
      predictions, labels = eval_pred.predictions, eval_pred.label_ids
      # If predictions are logits, convert to labels by taking the argmax
      predictions = np.argmax(predictions, axis=-1)
      
      scores = {}
      for scorer_name, scorer in self.scorers.items():
          scores[scorer_name] = scorer.compute(predictions, labels)

      return scores
    
   
    
    def get_vocab(self): 
        return self.vocab

    def evaluate_metrics(self, dataloader: DataLoader) -> dict:
      self.model.eval()
      labels = []
      predictions = []

      with tqdm(desc='Epoch %d - Evaluating' % self.epoch, unit='it', total=len(dataloader)) as pbar:
          for items in dataloader:
              items = {key: value.to(self.device) for key, value in items.items()}  # Move all items to the device
              input_ids = items["input_ids"]
              label = items["labels"]

              with torch.no_grad():  # Disable gradient computation during evaluation
                  logits = self.model(input_ids).logits

              output = logits.argmax(dim=-1).long()

              labels.append(label.cpu().numpy())
              predictions.append(output.cpu().numpy())

              pbar.update()

      predictions = np.concatenate(predictions)
      labels = np.concatenate(labels)

      eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

      scores = self.compute_scores(eval_pred)

      return scores

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

        # Using Trainer from transformers to simplify training
        training_args = TrainingArguments(
            output_dir=self.checkpoint_path,
            num_train_epochs=self.config.training.patience,
            per_device_train_batch_size=self.config.dataset.batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.training.learning_rate,
            load_best_model_at_end=True,
            logging_dir=os.path.join(self.checkpoint_path, "logs"),
            report_to=['none'], 
            save_total_limit=1  # Save only the best model
        )
        
      

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset,
            data_collator=collate_fn, 
            compute_metrics=self.compute_scores
        )


        # Start training
        trainer.train()

        # After training, evaluate on the test set
        test_results = trainer.evaluate(eval_dataset=self.test_dataset)
        self.logger.info("Test scores %s", test_results)

        # Save the results to files
        json.dump(test_results, open(os.path.join(self.checkpoint_path, "test_scores.json"), "w+"), ensure_ascii=False, indent=4)
        self.logger.info('Training completed.')

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
        # val_scores = self.evaluate_metrics(self.dev_dataloader)
        scores.append({
            # "val_scores": val_scores , 
            "test_scores": test_scores
        })
        with tqdm(desc='Epoch %d - Predicting' % self.epoch, unit='it', total=len(dataloader)) as pbar:
            for items in dataloader:
                items = items.to(self.device)
                input_ids = items.input_ids
                label = items.label
                _ , logits = self.model(input_ids, label)
                output = logits.argmax(dim=-1).long()
                
                labels.append(label[0].cpu().item())
                predictions.append(output[0].cpu().item())

                sentence = self.vocab.decode_sentence(input_ids)
                label = self.vocab.decode_label(label[0].cpu())
                prediction = self.vocab.decode_label(output.cpu())

                results.append({
                    "sentence": sentence,
                    "label": label,
                    "prediction": prediction
                })
                
                
                pbar.update()
           

        self.logger.info("Test scores %s", scores)
        json.dump(scores, open(os.path.join(self.checkpoint_path, "scores.json"), "w+"), ensure_ascii=False, indent=4)
        json.dump(results, open(os.path.join(self.checkpoint_path, "predictions.json"), "w+"), ensure_ascii=False, indent=4)
