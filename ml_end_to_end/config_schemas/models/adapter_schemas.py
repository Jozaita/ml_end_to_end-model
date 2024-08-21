from dataclasses import dataclass, field
from typing import Optional
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch import nn

from ml_end_to_end.utils.mixins import LoggableParamsMixin


@dataclass
class AdapterConfig(LoggableParamsMixin):
    _target_: str = MISSING

    def loggable_params(self) -> list[str]:
        return ["_target_"]

@dataclass
class MLPWithPoolingConfig(AdapterConfig):
    _target_: str = "ml_end_to_end.models.adapters.MLPWithPooling"
    output_feature_sizes:list[int] = MISSING
    biases:Optional[list[bool]] = None
    activation_fns:Optional[list[str]] = None
    dropouts_drop_probs:Optional[list[float]] = None
    batch_norms:Optional[list[bool]] = None
    order:str = "LABDN"
    standardize_input:bool=True
    pooling_method:Optional[str] = None
    output_attribute_to_use:Optional[str]= None

    def loggable_params(self) -> list[str]:
        return super().loggable_params() + ["output_feature_sizes",
                                            "biases",
                                            "activation_fns",
                                            "dropouts_drop_probs",
                                            "batch_norms",
                                            "order",
                                            "pooling_method",
                                            "output_attribute_to_use"]

@dataclass
class PoolerOutputAdapterConfig(MLPWithPoolingConfig):
    output_feature_sizes:list[int] = field(default_factory=lambda:[-1])
    output_attribute_to_use:str = "pooler_output"


def setup_config():
    cs = ConfigStore.instance()
    cs.store(name="mlp_with_pooling_schema",group="tasks/lightning_module/model/adapter",node=MLPWithPoolingConfig)
    
        
