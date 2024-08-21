


from dataclasses import dataclass

from omegaconf import MISSING, SI
from hydra.core.config_store import ConfigStore
from ml_end_to_end.config_schemas.base_schemas import TaskConfig
from ml_end_to_end.config_schemas.training import training_lightning_module_schemas
from ml_end_to_end.config_schemas.trainer import trainer_schemas
from ml_end_to_end.config_schemas import data_module_schemas


@dataclass
class TrainingTaskConfig(TaskConfig):
    best_training_checkpoint:str = SI("${infrastructure.mlflow.artifact_uri}/best-checkpoints/last.ckpt")
    last_training_checkpoint:str = SI("${infrastructure.mlflow.artifact_uri}/last-checkpoints/last.ckpt")

@dataclass
class CommonTrainingTaskConfig(TrainingTaskConfig):
    _target_:str = "ml_end_to_end.training.tasks.common_training_task.CommonTrainingTask"


@dataclass
class TarModelExportingTrainingTaskConfig(TrainingTaskConfig):
    tar_model_export_path:str = MISSING                

@dataclass
class DefaultCommonTrainingTaskConfig(TarModelExportingTrainingTaskConfig):
    _target_:str = "ml_end_to_end.training.tasks.tar_model_exporting_training_task.TarModelExportingTrainingTask"
    name: str = "binary_text_classification_task"
    data_module: data_module_schemas.DataModuleConfig = data_module_schemas.SrappedDataTextClassificationDataModuleConfig()
    lightning_module: training_lightning_module_schemas.TrainingLightningModuleConfig = training_lightning_module_schemas.CybuldeBinaryTextClassificationTrainingLightningModuleConfig()
    trainer: trainer_schemas.TrainerConfig = trainer_schemas.GPUDev()
    tar_model_export_path:str = SI("${infrastructure.mlflow.artifact_uri}/exported_model.tar.gz")

    


def setup_config():
    data_module_schemas.setup_config()
    trainer_schemas.setup_config()
    training_lightning_module_schemas.setup_config()

    cs = ConfigStore.instance()
    cs.store(name="common_training_task_schema", group="tasks", node=CommonTrainingTaskConfig)
    cs.store(name="test_task", node=DefaultCommonTrainingTaskConfig)