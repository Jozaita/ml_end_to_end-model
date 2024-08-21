from typing import Union,TYPE_CHECKING
from ml_end_to_end.config_schemas.config_schema import Config
from ml_end_to_end.data_modules.data_modules import DataModule, PartialDataModule
from ml_end_to_end.utils.io_utils import is_file
from ml_end_to_end.utils.mlflow_utils import activate_mlflow, log_artifacts_for_reproducibility
from training.lightning_modules.bases import TrainingLightningModule
from training.tasks.bases import TrainingTask
from lightning.pytorch import Trainer
if TYPE_CHECKING:
    from ml_end_to_end.config_schemas.config_schema import Config
    from ml_end_to_end.config_schemas.training.training_task_schemas import TrainingTaskConfig


class CommonTrainingTask(TrainingTask):
    def __init__(self,
                 name:str,
                 data_module:Union[DataModule,PartialDataModule],
                 lightning_module:TrainingLightningModule,
                 trainer:Trainer,
                 best_training_checkpoint:str,
                 last_training_checkpoint:str)->None:
        super().__init__(name=name,
                         data_module=data_module,
                         lightning_module=lightning_module,
                         trainer=trainer,
                         best_training_checkpoint=best_training_checkpoint,
                         last_training_checkpoint=last_training_checkpoint)
        
    def run(self,config:"Config",task_config:"TrainingTaskConfig"):
        experiment_name = config.infrastructure.mlflow.experiment_name
        run_id = config.infrastructure.mlflow.run_id
        run_name = config.infrastructure.mlflow.run_name
        with activate_mlflow(experiment_name=experiment_name,
                             run_id=run_id,
                             run_name=run_name) as _:
            if self.trainer.is_global_zero:
                log_artifacts_for_reproducibility()
                #log_training_hparams(config)
            if is_file(self.last_training_checkpoint):
                self.logger.info(f"Found checkpoint!:{self.last_training_checkpoint}. Resuming training...")
                assert isinstance(self.data_module,DataModule)
                self.trainer.fit(model=self.lightning_module,datamodule=self.data_module,ckpt_path=self.last_training_checkpoint)
            else:
                self.trainer.fit(model=self.lightning_module,datamodule=self.data_module)

            self.logger.info("Training finished...")

