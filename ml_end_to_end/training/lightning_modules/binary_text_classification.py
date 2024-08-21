

from typing import Optional
from collections import defaultdict
import torch
import mlflow
from ml_end_to_end.models.transformations import Transformation
from ml_end_to_end.models.adapters import Adapter
from ml_end_to_end.models.models import Model
from ml_end_to_end.utils.torch_utils import plot_confusion_matrix
from training.lightning_modules.bases import ModelStateDictExportingTrainingLightningModule, PartialOptimizerType, TrainingLightningModule
from training.loss_functions import LossFunction
from training.schedulers import LightningScheduler
from torchmetrics.classification import BinaryAccuracy,BinaryF1Score,BinaryConfusionMatrix
from transformers import BatchEncoding
from torch import Tensor

class BinaryTextClassificationLightningModule(ModelStateDictExportingTrainingLightningModule):
    def __init__(self,
                 model:Model,
                 loss: LossFunction,
                 optimizer: PartialOptimizerType,
                 scheduler: Optional[LightningScheduler])->None:
        super().__init__(model=model, loss=loss, optimizer=optimizer, scheduler=scheduler)

        self.training_accuracy = BinaryAccuracy()
        self.validation_accuracy = BinaryAccuracy()

        self.training_f1_score = BinaryF1Score()
        self.validation_f1_score = BinaryF1Score()

        self.training_confusion_matrix = BinaryConfusionMatrix()
        self.validation_confusion_matrix = BinaryConfusionMatrix()

        self.train_step_output = defaultdict(list)
        self.validation_step_output = defaultdict(list)

        self.pos_weight:Optional[Tensor] = None

    def set_pos_weight(self, pos_weight:Tensor) -> None:
        self.pos_weight = pos_weight


    def forward(self,texts:BatchEncoding)->Tensor:
        return self.model(texts)
    
    def training_step(self, batch:tuple[BatchEncoding,Tensor], batch_idx:int)->Tensor:
        texts,labels = batch
        logits = self(texts)
        self.pos_weight = self.pos_weight.to(self.device)
        loss = self.loss(logits, labels, self.pos_weight)
        self.log('train_loss', loss,sync_dist=True)

        self.training_accuracy(logits, labels)
        self.training_f1_score(logits, labels)

        self.log('train_f1_score',self.training_f1_score,on_step=False,on_epoch=True)
        self.log('train_accuracy', self.training_accuracy,on_step=False,on_epoch=True)

        self.train_step_output["logits"].append(logits)
        self.train_step_output["labels"].append(labels)

        return loss
    
    def on_train_epoch_end(self)->None:
        all_logits = torch.stack(self.train_step_output["logits"])
        all_labels = torch.stack(self.train_step_output["labels"])

        confusion_matrix = self.training_confusion_matrix(all_logits,all_labels)
        figure = plot_confusion_matrix(confusion_matrix,["0","1"])
        mlflow.log_figure(figure, "train_confusion_matrix.png")

        self.train_step_output = defaultdict(list)


    def validation_step(self, batch:tuple[BatchEncoding,Tensor], batch_idx:int)->Tensor:
        texts,labels = batch
        logits = self(texts)
        loss = self.loss(logits, labels)
        self.log('val_loss', loss,sync_dist=True)

        self.validation_accuracy(logits, labels)
        self.validation_f1_score(logits, labels)

        self.log('val_f1_score',self.validation_f1_score,on_step=False,on_epoch=True)
        self.log('val_accuracy', self.validation_accuracy,on_step=False,on_epoch=True)

        self.validation_step_output["logits"].append(logits)
        self.validation_step_output["labels"].append(labels)

        return {"loss":loss, "predictions":logits, "labels":labels}
    
    def on_validation_epoch_end(self)->None:
        all_logits = torch.stack(self.validation_step_output["logits"])
        all_labels = torch.stack(self.validation_step_output["labels"])

        confusion_matrix = self.validation_confusion_matrix(all_logits, all_labels)
        figure = plot_confusion_matrix(confusion_matrix,["0","1"])
        #mlflow.log_figure(figure)
        mlflow.log_figure(figure, "validation_confusion_matrix.png")

        self.validation_step_output = defaultdict(list)

    
    def get_transformation(self) -> Transformation:
        return self.model.get_transformation()
    
    def export_model_state_dict(self, checkpoint_path:str)->str:
        return self.common_export_model_state_dict(checkpoint_path)