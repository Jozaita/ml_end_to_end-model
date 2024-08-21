from typing import Union,TYPE_CHECKING
from ml_end_to_end.config_schemas.config_schema import Config
from ml_end_to_end.data_modules.data_modules import DataModule, PartialDataModule
from ml_end_to_end.evaluation.lightning_modules.bases import PartialEvaluationLightningModule
from ml_end_to_end.evaluation.tasks.bases import TarModelEvaluationTask
from ml_end_to_end.utils.io_utils import is_file
from ml_end_to_end.utils.mlflow_utils import activate_mlflow, log_artifacts_for_reproducibility, log_model
from training.lightning_modules.bases import TrainingLightningModule
from training.tasks.bases import TrainingTask
from lightning.pytorch import Trainer
if TYPE_CHECKING:
    from ml_end_to_end.config_schemas.config_schema import Config
    from ml_end_to_end.config_schemas.training.training_task_schemas import TrainingTaskConfig

from hydra.utils import instantiate
class CommonEvaluationTask(TarModelEvaluationTask):
    def __init__(self,
                 name:str,
                 data_module:Union[DataModule,PartialDataModule],
                 lightning_module:PartialEvaluationLightningModule,
                 trainer:Trainer,
                 tar_model_path:str)->None:
        super().__init__(name=name,
                         data_module=data_module,
                         lightning_module=lightning_module,
                         trainer=trainer,
                         tar_model_path=tar_model_path)
        
    def run(self,config:"Config",task_config:"EvaluationTaskConfig")->None:
        experiment_name = config.infrastructure.mlflow.experiment_name
        run_id = config.infrastructure.mlflow.run_id
        run_name = config.infrastructure.mlflow.run_name
        with activate_mlflow(experiment_name=experiment_name,
                             run_id=run_id,
                             run_name=run_name) as _:
            self.trainer.test(model=self.lightning_module,datamodule=self.data_module)

        model_selector = instantiate(config.model_selector)
        if model_selector is not None:
            if model_selector.is_selected():
                log_model(
                    config.infrastructure.mlflow,
                    model_selector.get_new_best_run_tag(),
                    config.registered_model_name
                )

