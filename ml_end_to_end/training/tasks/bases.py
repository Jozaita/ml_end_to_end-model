

from abc import ABC, abstractmethod
from functools import partial
from typing import Union,TYPE_CHECKING

from lightning.pytorch import Trainer

from ml_end_to_end.data_modules.data_modules import DataModule, PartialDataModule
from ml_end_to_end.utils.utils import get_logger
from training.lightning_modules.bases import TrainingLightningModule

if TYPE_CHECKING:
    from ml_end_to_end.config_schemas.config_schema import Config
    from ml_end_to_end.config_schemas.training.training_task_schemas import TrainingTaskConfig

class TrainingTask(ABC):
    def __init__(self,
                 name:str,
                 data_module:Union[DataModule,PartialDataModule],
                 lightning_module:TrainingLightningModule,
                 trainer:Trainer,
                 best_training_checkpoint:str,
                 last_training_checkpoint:str)->None:
        super().__init__()

        self.name = name
        self.trainer = trainer
        self.best_training_checkpoint = best_training_checkpoint
        self.last_training_checkpoint = last_training_checkpoint
        self.logger = get_logger(self.__class__.__name__)

        self.lightning_module = lightning_module

        if isinstance(data_module,partial):
            transformation = self.lightning_module.get_transformation()
            self.data_module = data_module(transformation=transformation)   
        else:
            self.data_module = data_module


    @abstractmethod
    def run(self,config:"Config",task_config:"TrainingTaskConfig"):
        ...


        


