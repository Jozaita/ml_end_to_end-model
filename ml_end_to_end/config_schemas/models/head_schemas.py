from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from ml_end_to_end.utils.mixins import LoggableParamsMixin

@dataclass
class HeadConfig(LoggableParamsMixin):
    _target_: str = MISSING

    def loggable_params(self) -> list[str]:
        return ["_target_"]

@dataclass
class SigmoidHeadConfig(HeadConfig):
    _target_: str = "ml_end_to_end.models.heads.SigmoidHead"
    in_features:int = MISSING
    out_features:int = MISSING


@dataclass
class BinaryClassificationSigmoidHeadConfig(SigmoidHeadConfig):
    in_features:int = 128
    out_features:int = 1


def setup_config():
    cs = ConfigStore.instance()
    cs.store(name="sigmoid_head_schema",group="tasks/lightning_module/model/head",node=SigmoidHeadConfig)
    
        
