from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from ml_end_to_end.config_schemas.models.backbone_schemas import BackBoneConfig, BertTinyHuggingFaceBackboneConfig
from ml_end_to_end.config_schemas.models.adapter_schemas import AdapterConfig, PoolerOutputAdapterConfig
from ml_end_to_end.config_schemas.models.head_schemas import BinaryClassificationSigmoidHeadConfig, HeadConfig
from typing import Optional

from ml_end_to_end.utils.mixins import LoggableParamsMixin

@dataclass
class ModelConfig(LoggableParamsMixin):
    _target_: str = MISSING

    def loggable_params(self) -> list[str]:
        return ["_target_"]

@dataclass
class BinaryTextClassificationModelConfig(ModelConfig):
    _target_: str = "ml_end_to_end.models.models.BinaryTextClassificationModel"
    backbone:BackBoneConfig = MISSING
    adapter:Optional[AdapterConfig] = None
    head: HeadConfig = MISSING

@dataclass
class TinyBinaryTextClassificationModelConfig(BinaryTextClassificationModelConfig):
    backbone:BackBoneConfig = BertTinyHuggingFaceBackboneConfig()
    adapter:AdapterConfig = PoolerOutputAdapterConfig()
    head: HeadConfig = BinaryClassificationSigmoidHeadConfig()

def setup_config():
    cs = ConfigStore.instance()
    cs.store(name="binary_text_classification_model_schema",group="tasks/lightning_module/model",node=BinaryTextClassificationModelConfig)
    cs.store(name="test_model",node=TinyBinaryTextClassificationModelConfig)
        
        
