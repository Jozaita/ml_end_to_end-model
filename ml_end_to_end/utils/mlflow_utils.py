

import dataclasses
from typing import TYPE_CHECKING, Any, Generator
from contextlib import contextmanager
import os
from ml_end_to_end.config_schemas.infrastructure.infrastructure_schema import MLFlowConfig
from ml_end_to_end.utils.mixins import LoggableParamsMixin
import mlflow
from typing import Iterable, Optional
from mlflow.pyfunc import PythonModel
if TYPE_CHECKING:
    from ml_end_to_end.config_schemas.config_schema import Config


MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

@contextmanager
def activate_mlflow(experiment_name:Optional[str] = None,
                    run_id:Optional[str] = None,
                    run_name:Optional[str] = None,
                    )->Iterable[mlflow.ActiveRun]:
    
    set_experiment(experiment_name)

    with mlflow.start_run(run_id=run_id, run_name=run_name) as run:
        yield run



def set_experiment(experiment_name:Optional[str] = None)->None:
    if experiment_name is None:
        experiment_name = "default"

    try: 
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.RestException:
        pass

    mlflow.set_experiment(experiment_name)
    

def log_artifacts_for_reproducibility()->None:
    locations_to_store = [
        "./ml_end_to_end",
        "./docker",
        "./pyproject.toml",
        "./poetry.lock"
    ]

    for location in locations_to_store:
        mlflow.log_artifact(location,"reproduction")





def log_training_hparams(config: "Config") -> None:
    logged_nodes = set()

    def loggable_params(node: Any, path: list[str]) -> Generator[tuple[str, Any], None, None]:
        if isinstance(node, LoggableParamsMixin) and id(node) not in logged_nodes:
            for param_name in node.loggable_params():
                yield ".".join(path + [param_name]), getattr(node, param_name)
            logged_nodes.add(id(node))
        children = None
        if isinstance(node, dict):
            children = node.items()
        if dataclasses.is_dataclass(node):
            children = ((f.name, getattr(node, f.name)) for f in dataclasses.fields(node))  # type: ignore

        if children is None:
            return
        for key, val in children:
            for item in loggable_params(val, path + [key]):
                yield item

    params = dict(loggable_params(config, []))
    mlflow.log_params(params)

def get_client() -> mlflow.tracking.MlflowClient:
    return mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_all_experiment_ids() ->list[str]:
    return [exp.experiment_id for exp in mlflow.search_experiments()]

def get_best_run()->dict[str,Any]:
    best_runs = mlflow.search_runs(get_all_experiment_ids(),filter_string="tag.best_run LIKE 'v%'")
    if len(best_runs) == 0:
        return {}
    
    indices = best_runs["tags.best_run"].str.split("v").str[-1].astype(int).sort_values()
    best_runs = best_runs.reindex(index=indices.index)
    return best_runs.iloc[-1].to_dict()


class DummyWrapper(PythonModel):
    def load_context(self, context:str)->None:
        pass
    def predict(self, context: Any, model_input: Any)->Any:
        pass

def log_model(
        mlflow_config:MLFlowConfig,
        new_best_run_tag:str,
        registered_model_name: str
)-> None:
    experiment_name = mlflow_config.experiment_name
    run_id = mlflow_config.run_id
    run_name = mlflow_config.run_name

    with activate_mlflow(experiment_name=experiment_name, run_id=run_id, run_name=run_name) as run:
        mlflow.pyfunc.log_model(
            artifact_path="",
            python_model=DummyWrapper(),
            registered_model_name=registered_model_name
        )

        mlflow.set_tag("best_run",new_best_run_tag)
        