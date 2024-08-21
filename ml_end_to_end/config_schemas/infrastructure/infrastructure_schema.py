

from dataclasses import dataclass
from typing import Optional
from ml_end_to_end.config_schemas.infrastructure.instance_group_creator_schemas import InstanceGroupCreatorConfig
from omegaconf import SI
from hydra.core.config_store import ConfigStore

@dataclass
class MLFlowConfig:
    mlflow_external_tracking_uri: str = SI("${oc.env:MLFLOW_TRACKING_URI,localhost:6101}")
    mlflow_internal_tracking_uri: str = SI("${oc.env:MLFLOW_INTERNAL_TRACKING_URI,localhost:6101}")
    experiment_name: str = "Default"
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_url: str = SI("${.mlflow_external_tracking_uri}/#/experiments/${.experiment_id}/runs/${.run_id}")
    artifact_uri: Optional[str] = None



@dataclass 
class InfrastructureConfig:
    project_id: str = "ageless-fire-423616-e1"
    zone: str = "europe-west1-b"
    region: str = "europe-west1"
    instance_group_creator: InstanceGroupCreatorConfig = InstanceGroupCreatorConfig()
    mlflow: MLFlowConfig = MLFlowConfig()
    etcd_ip: Optional[str] = "10.164.0.12:2379"
    


def setup_config():
    cs = ConfigStore.instance()
    cs.store(name="infrastructure_schema",group="infrastructure", node=InfrastructureConfig)