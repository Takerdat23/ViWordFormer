from torch import Tensor
from torch.utils.data import DataLoader
import os
from shutil import copyfile
import numpy as np
from tqdm import tqdm
import json
from builders.task_builder import META_TASK
from builders.dataset_builder import build_dataset
from tasks.base_task import BaseTask
from data_utils import collate_fn
from evaluation import F1, Precision, Recall

@META_TASK.register()
class Transformer_ABSA_Task(BaseTask):
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

    def collate_with_pad(self, items):
        return collate_fn(items, self.vocab.pad_idx)

    def create_dataloaders(self, config):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=self.collate_with_pad
        )
        self.dev_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=self.collate_with_pad
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            collate_fn=self.collate_with_pad
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
        all_aspect_label = []
        all_aspect_pred = []
        all_sentiment_label = []
        all_sentiment_pred = []
      
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

                
                all_aspect_pred.append((output != 0).long().cpu().numpy())  # Binary aspect presence (1 if sentiment != 0)
                all_aspect_label.append((label != 0).long().cpu().numpy())   # Binary aspect ground truth (1 if sentiment != 0)

                
                all_sentiment_label.append(label.cpu().numpy())  # Store the true labels
                all_sentiment_pred.append(output.cpu().numpy())  # Store the predictions
                
                pbar.update()
                
       
        # Convert lists to numpy arrays for easier indexing and computation
        #Aspect
        all_aspect_label = np.concatenate(all_aspect_label, axis=0)  
        all_aspect_pred = np.concatenate(all_aspect_pred, axis=0)  
        #Sentiment
        all_sentiment_label = np.concatenate(all_sentiment_label, axis=0)  
        all_sentiment_pred = np.concatenate(all_sentiment_pred, axis=0)  
        # Calculate accuracy for each aspect
        for i, aspect in enumerate(aspect_list):
            preds_aspect = all_sentiment_pred[:, i]
            labels_aspect = all_sentiment_label[:, i]
            
            # Compute additional metrics using the compute_scores function
            aspect_scores = self.compute_scores(preds_aspect, labels_aspect)
            
            aspect_scores = {metric: float(value) if isinstance(value, np.float64) else value for metric, value in aspect_scores.items()}
            aspect_wise_scores[aspect] = aspect_scores
        
            # Accumulate scores to compute averages later
            total_f1 += aspect_scores['f1']
            total_recall += aspect_scores['recall']
            total_precision += aspect_scores['precision']
   
        aspect_score = self.compute_scores(all_aspect_pred.flatten(), all_aspect_label.flatten())
        aspect_score = {metric: float(value) if isinstance(value, np.float64) else value for metric, value in aspect_score.items()}
        Sentiment_score = self.compute_scores(all_sentiment_pred.flatten() ,all_sentiment_label.flatten())
        Sentiment_score = {metric: float(value) if isinstance(value, np.float64) else value for metric, value in Sentiment_score.items()}
    
    

        return {
        'aspect':aspect_score,
        'sentiment': Sentiment_score,
        # 'aspect_wise_scores': aspect_wise_scores
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
        scores = self.evaluate_metrics(self.test_dataloader)
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
            scores = self.evaluate_metrics(self.dev_dataloader)
            self.logger.info("Validation scores %s", scores)
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
        
    
