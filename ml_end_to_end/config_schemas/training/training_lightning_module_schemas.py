from omegaconf import MISSING
from dataclasses import dataclass
from typing import Optional
from hydra.core.config_store import ConfigStore
from ml_end_to_end.config_schemas.models.model_schemas import ModelConfig, TinyBinaryTextClassificationModelConfig
from ml_end_to_end.config_schemas.base_schemas import LightningModuleConfig
from ml_end_to_end.config_schemas.training.loss_schemas import BCEWithLogitsLossConfig, LossFunctionConfig
from ml_end_to_end.config_schemas.training.optimizer_schemas import AdamWOptimizerConfig, OptimizerConfig
from ml_end_to_end.config_schemas.training.scheduler_schemas import ReduceLROnPlateauLightningSchedulerConfig, LightningSchedulerConfig
from ml_end_to_end.utils.mixins import LoggableParamsMixin


@dataclass
class TrainingLightningModuleConfig(LightningModuleConfig,LoggableParamsMixin):
    _target_:str = MISSING
    model: ModelConfig = MISSING
    loss: LossFunctionConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    scheduler: Optional[LightningSchedulerConfig] = None

    def loggable_params(self) -> list[str]:
        return ["_target_"]

@dataclass
class BinaryTextClassificationTrainingLightningModuleConfig(TrainingLightningModuleConfig):
    _target_:str = "ml_end_to_end.training.lightning_modules.binary_text_classification.BinaryTextClassificationLightningModule"

@dataclass
class CybuldeBinaryTextClassificationTrainingLightningModuleConfig(BinaryTextClassificationTrainingLightningModuleConfig):
    model: ModelConfig = TinyBinaryTextClassificationModelConfig()
    loss: LossFunctionConfig = BCEWithLogitsLossConfig()
    optimizer: OptimizerConfig = AdamWOptimizerConfig()
    scheduler: Optional[LightningSchedulerConfig] = ReduceLROnPlateauLightningSchedulerConfig()



def setup_config():
    cs = ConfigStore.instance()
    cs.store(name="binary_text_classification_training_lightning_module_schema", group="tasks/lightning_module", node=BinaryTextClassificationTrainingLightningModuleConfig)
    cs.store(name="lm_test", node=CybuldeBinaryTextClassificationTrainingLightningModuleConfig)