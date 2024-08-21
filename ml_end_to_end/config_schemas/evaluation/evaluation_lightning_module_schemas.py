


from dataclasses import dataclass

from omegaconf import MISSING

from ml_end_to_end.config_schemas.base_schemas import LightningModuleConfig


@dataclass
class EvaluationLightningModuleConfig(LightningModuleConfig):
    _target_: str = MISSING
    _partial_: bool = False

    def loggable_params(self) -> list[str]:
        return ["_target_"]

    
@dataclass
class PartialEvaluationLightningModuleConfig(EvaluationLightningModuleConfig):
    _partial_:bool = True


@dataclass
class BinaryTextEvaluationLightningModuleConfig(PartialEvaluationLightningModuleConfig):
    _target_: str = "ml_end_to_end.evaluation.lightning_modules.binary_text_evaluation.BinaryTextEvaluationLightningModule"