from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from ml_end_to_end.utils.mixins import LoggableParamsMixin
@dataclass
class LossFunctionConfig:
    _target_: str = MISSING

@dataclass
class BCEWithLogitsLossConfig(LossFunctionConfig,LoggableParamsMixin):
    _target_: str = "ml_end_to_end.training.loss_functions.BCEWithLogitsLoss"
    reduction: str = "mean"

    def loggable_params(self) -> list[str]:
        return ["_target_"]


def setup_config():
    cs = ConfigStore.instance()
    cs.store(name="bce_with_logits_loss_schema", group="tasks/lightning_module/loss_function", node=BCEWithLogitsLossConfig)
