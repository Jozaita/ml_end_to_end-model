

from typing import Optional
from omegaconf import MISSING
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from ml_end_to_end.utils.mixins import LoggableParamsMixin

@dataclass
class OptimizerConfig(LoggableParamsMixin):
    _target_: str = MISSING
    _partial_: bool = True
    lr:float = MISSING

    def loggable_params(self) -> list[str]:
        return ["_target_","lr"]

@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    _target_:str = "torch.optim.Adam"
    lr:float = 5e-5
    weight_decay:float = 1e-3
    betas: tuple[float,float] = (0.9,0.999)
    eps:float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    foreach: Optional[bool] = False
    maximize: bool = False
    capturable: bool = False

@dataclass
class AdamWOptimizerConfig(AdamOptimizerConfig):
    _target_: str = "torch.optim.AdamW"



def setup_config():
    cs = ConfigStore.instance()
    cs.store(name="adam_optimizer_schema", group="tasks/lightning_module/optimizers", node=AdamOptimizerConfig)
    cs.store(name="adamw_optimizer_schema", group="tasks/lightning_module/optimizers", node=AdamWOptimizerConfig)
