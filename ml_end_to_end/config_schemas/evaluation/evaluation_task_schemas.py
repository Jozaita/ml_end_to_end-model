

from dataclasses import dataclass
from typing import Union

from omegaconf import MISSING
from ml_end_to_end.config_schemas.evaluation.evaluation_lightning_module_schemas import BinaryTextEvaluationLightningModuleConfig, EvaluationLightningModuleConfig, PartialEvaluationLightningModuleConfig
from ml_end_to_end.config_schemas.base_schemas import TaskConfig


@dataclass
class EvaluationTaskConfig(TaskConfig):
    pass


@dataclass
class TarModelEvaluationTaskConfig(EvaluationTaskConfig):
    tar_model_path: str = MISSING
    lightning_module:PartialEvaluationLightningModuleConfig


@dataclass
class CommonEvaluationTaskConfig(TarModelEvaluationTaskConfig):
    _target_: str = "ml_end_to_end.evaluation.tasks.common_evaluation_task.CommonEvaluationTask"

@dataclass
class DefaultEvaluationTaskConfig(CommonEvaluationTaskConfig):
    name: str = "binary_text_evaluation_task"
    lightning_module: PartialEvaluationLightningModuleConfig = BinaryTextEvaluationLightningModuleConfig()
