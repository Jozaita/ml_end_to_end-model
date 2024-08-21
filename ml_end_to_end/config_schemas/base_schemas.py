from dataclasses import dataclass
from omegaconf import MISSING

from ml_end_to_end.config_schemas.data_module_schemas import (DataModuleConfig)
from ml_end_to_end.config_schemas.trainer.trainer_schemas import TrainerConfig
from ml_end_to_end.utils.mixins import LoggableParamsMixin


@dataclass
class LightningModuleConfig(LoggableParamsMixin):
    _target_: str = MISSING


@dataclass
class TaskConfig(LoggableParamsMixin):
    _target_:str = MISSING
    name:str = MISSING
    data_module:DataModuleConfig = MISSING
    lightning_module:LightningModuleConfig = MISSING 
    trainer:TrainerConfig = MISSING

    def loggable_params(self) -> list[str]:
        return ["_target_"]