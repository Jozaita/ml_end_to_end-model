

from dataclasses import dataclass
from omegaconf import MISSING
from typing import Optional
from hydra.core.config_store import ConfigStore

from ml_end_to_end.utils.mixins import LoggableParamsMixin

@dataclass
class SchedulerConfig(LoggableParamsMixin):
    _target_:str = MISSING
    _partial_: bool = True
    def loggable_params(self) -> list[str]:
        return ["_target_"]

@dataclass
class ReduceLROnPlateauSchedulerConfig(SchedulerConfig):
    _target_:str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    mode:str = "max"
    factor:float = 0.1
    patience: int = 10
    threshold:float = 1e-4
    threshold_mode:str = "rel"
    cooldown: int = 0
    min_lr: float = 0
    eps: float = 1e-8
    verbose: bool = False


@dataclass
class LightningSchedulerConfig:
    _target_:str = MISSING
    scheduler: SchedulerConfig = MISSING
    interval: str = "epoch"
    frequency: int = 1
    monitor: str = "val_f1_score"
    strict: bool = True
    name: Optional[str] = None

@dataclass
class CommonLightiningSchedulerConfig(LightningSchedulerConfig):
    _target_:str = "ml_end_to_end.training.schedulers.CommonLightingScheduler"

@dataclass
class ReduceLROnPlateauLightningSchedulerConfig(CommonLightiningSchedulerConfig):
    scheduler: SchedulerConfig = ReduceLROnPlateauSchedulerConfig()
    


def setup_config():
    cs = ConfigStore.instance()
    cs.store(name="reduce_lr_on_plateau_scheduler_schema", group="tasks/lightning_module/scheduler", node=ReduceLROnPlateauLightningSchedulerConfig)