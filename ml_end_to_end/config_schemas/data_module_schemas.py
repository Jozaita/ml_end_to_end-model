from dataclasses import dataclass
from typing import Any
from omegaconf import MISSING, SI
from hydra.core.config_store import ConfigStore
from ml_end_to_end.config_schemas.models import transformation_schemas
from ml_end_to_end.utils.mixins import LoggableParamsMixin
@dataclass
class DataModuleConfig(LoggableParamsMixin):
    _target_:str = MISSING
    batch_size:int = MISSING
    shuffle:bool = False
    num_workers:int = 8 
    pin_memory:bool = True 
    drop_last:bool = True 
    persistent_workers:bool = False

    def loggable_params(self) -> list[str]:
        return ["_target_", "batch_size"]

@dataclass
class TextClassificationDataModuleConfig(DataModuleConfig):
    _target_:str = 'ml_end_to_end.data_modules.data_modules.TextClassificationDataModule'
    train_df_path:str = MISSING
    dev_df_path:str = MISSING
    test_df_path:str = MISSING
    transformation:transformation_schemas.TransformationConfig = MISSING
    text_column_name:str = "cleaned_text"
    label_column_name:str = "label"

@dataclass
class SrappedDataTextClassificationDataModuleConfig(TextClassificationDataModuleConfig):
    batch_size:int = 64
    train_df_path:str = "gs://ml_end_to_end/data/processed/rebalanced_split/train.parquet"
    dev_df_path:str = "gs://ml_end_to_end/data/processed/rebalanced_split/dev.parquet"
    test_df_path:str = "gs://ml_end_to_end/data/processed/rebalanced_split/test.parquet"
    transformation:transformation_schemas.TransformationConfig = SI("${..lightning_module.model.backbone.transformation}")


def setup_config():
    transformation_schemas.setup_config()
    cs = ConfigStore.instance()
    cs.store(name="text_classification_data_module_schema",group="tasks/data_module",node=TextClassificationDataModuleConfig)
  
        
