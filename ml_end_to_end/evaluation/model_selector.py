
from typing import Any
from mlflow.entities import Run

from ml_end_to_end.utils.mlflow_utils import get_best_run, get_client
from ml_end_to_end.utils.utils import get_logger

class MetricComparer:
    def __init__(self,
                 bigger_is_better:bool,
                 can_be_equal:bool,
                 metric_name:str,
                 threshold:float=0.0) -> None:
        
        super().__init__()

        self.bigger_is_better = bigger_is_better
        self.can_be_equal = can_be_equal
        self.metric_name = metric_name
        self.threshold = threshold
    
    def get_current_metric_value(self,run:Run)->float:
        current_metric_value = run.data.metrics.get(self.metric_name,None)
        if current_metric_value is None:
            raise RuntimeError(f"No metric value found for metric '{self.metric_name}' in run {run.info.run_id}")
        return current_metric_value

    def is_metric_better(self,run:Run,best_run_data:dict[str,Any])->bool:
        if not best_run_data:
            return True
        
        current_metric_value = self.get_current_metric_value(run)
        best_metric_value = best_run_data[f"metrics.{self.metric_name}"]
        
        if self.can_be_equal and current_metric_value == best_metric_value:
            return True
        if self.bigger_is_better:
            current_metric_value -= self.threshold
            return current_metric_value>best_metric_value
        else:
            current_metric_value += self.threshold
            return current_metric_value<best_metric_value


    

class ModelSelector:
    def __init__(self,
                 mlflow_run_id:str,
                 must_be_better_metric_comparers:dict[str,MetricComparer] = {},
                 to_be_thresholded_metric_comparers:dict[str,MetricComparer] = {},
                 threshold:float = 0.0
                 ) -> None:
        if not(must_be_better_metric_comparers) and not(to_be_thresholded_metric_comparers):
            raise ValueError("'must_be_better_metric_comparers' and 'to_be_thresholded_metric_comparers' must be set")
        self.mlflow_run_id = mlflow_run_id
        self.must_be_better_metric_comparers = must_be_better_metric_comparers
        self.to_be_thresholded_metric_comparers = to_be_thresholded_metric_comparers
        self.threshold = threshold

        self.logger = get_logger(self.__class__.__name__)

        client = get_client()

        self.run = client.get_run(self.mlflow_run_id)
        self.best_run_data = get_best_run()
        self.new_best_run_tag =  None

    def is_selected(self)->bool:
        is_selected = self._is_selected(self.run)
        if is_selected:
            self.new_best_run_tag = self.get_new_best_run_tag()
        return is_selected

    def _is_selected(self,run:Run)->bool:
        for metric_name, metric_comparer in self.must_be_better_metric_comparers.items():
            if not metric_comparer.is_metric_better(run, self.best_run_data):
                self.logger.info(f"Metric '{metric_name}' did not meet the required threshold")
                return False

        hits = []
        for metric_comparer in self.to_be_thresholded_metric_comparers.values():
            is_better =  metric_comparer.is_metric_better(run, self.best_run_data)
            hits.append(int(is_better))

        if not hits:
            return True
        
        mean_hits = sum(hits)/len(hits)

        return mean_hits > self.threshold

    def get_new_best_run_tag(self)->str:
        if len(self.best_run_data) == 0:
            return "v1"
        last_tag:str = self.best_run_data["tags.best_run"]
        last_version = int(last_tag[1:])
        return f"v{last_version+1}"
    






