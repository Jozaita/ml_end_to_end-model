

from typing import Optional
from collections import defaultdict
import torch
from ml_end_to_end.evaluation.lightning_modules.bases import EvaluationLightningModule
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

class BinaryTextEvaluationLightningModule(EvaluationLightningModule):
    def __init__(self,
                 model:Model)->None:
        super().__init__(model=model)

        self.test_accuracy = BinaryAccuracy()


        self.test_f1_score = BinaryF1Score()


        self.test_confusion_matrix = BinaryConfusionMatrix()


        self.test_step_output = defaultdict(list)


    def forward(self,texts:BatchEncoding)->Tensor:
        return self.model(texts)
    
    def test_step(self, batch:tuple[BatchEncoding,Tensor], batch_idx:int)->Tensor:
        texts,labels = batch
        logits = self(texts)


        self.test_accuracy(logits, labels)
        self.test_f1_score(logits, labels)

        self.log('test_f1_score',self.test_f1_score,on_step=False,on_epoch=True)
        self.log('test_accuracy', self.test_accuracy,on_step=False,on_epoch=True)

        self.test_step_output["logits"].append(logits)
        self.test_step_output["labels"].append(labels)

    
    def on_test_epoch_end(self)->None:
        all_logits = torch.stack(self.test_step_output["logits"])
        all_labels = torch.stack(self.test_step_output["labels"])

        confusion_matrix = self.test_confusion_matrix(all_logits,all_labels)
        figure = plot_confusion_matrix(confusion_matrix,["0","1"])
        mlflow.log_figure(figure, "test_confusion_matrix.png")

        self.test_step_output = defaultdict(list)


    
    def get_transformation(self) -> Transformation:
        return self.model.get_transformation()
    
