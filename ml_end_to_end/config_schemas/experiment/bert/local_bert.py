


from dataclasses import dataclass, field
from typing import Optional
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from ml_end_to_end.config_schemas.config_schema import Config

from ml_end_to_end.config_schemas.evaluation.evaluation_task_schemas import CommonEvaluationTaskConfig, DefaultEvaluationTaskConfig
from ml_end_to_end.config_schemas.base_schemas import TaskConfig
from ml_end_to_end.config_schemas.evaluation.model_selector_schemas import CyberBullyingModelSelectorConfig, ModelSelectorConfig
from ml_end_to_end.config_schemas.trainer.trainer_schemas import GPUProd
from ml_end_to_end.config_schemas.training.training_task_schemas import DefaultCommonTrainingTaskConfig


@dataclass
class LocalBertExperiment(Config):
    tasks:dict[str,TaskConfig] = field(default_factory=lambda:{
        "binary_text_classification_task":DefaultCommonTrainingTaskConfig(trainer=GPUProd()),
        "binary_text_evaluation_task":DefaultEvaluationTaskConfig(),
    })
    model_selector:Optional[ModelSelectorConfig] = CyberBullyingModelSelectorConfig()
    registered_model_name:Optional[str] = "bert_tiny"


FinalLocalBertExperiment = OmegaConf.merge(LocalBertExperiment,
                                           OmegaConf.from_dotlist([
                                               "infrastructure.mlflow.experiment_name=ml_end_to_end",
                                               "tasks.binary_text_classification_task.data_module.batch_size=1024",
                                               "tasks.binary_text_evaluation_task.tar_model_path=${tasks.binary_text_classification_task.tar_model_export_path}",
                                               "tasks.binary_text_evaluation_task.data_module=${tasks.binary_text_classification_task.data_module}",
                                               "tasks.binary_text_evaluation_task.trainer=${tasks.binary_text_classification_task.trainer}"
                                           ]))

cs = ConfigStore.instance()
cs.store(name="local_bert",group="experiment/bert",node=FinalLocalBertExperiment,package="_global_")