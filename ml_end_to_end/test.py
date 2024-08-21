import hydra
from omegaconf import DictConfig,OmegaConf
from ml_end_to_end.config_schemas.training.training_task_schemas import setup_config
from hydra.utils import instantiate
from ml_end_to_end.utils.mlflow_utils import get_all_experiment_ids,get_best_run
from ml_end_to_end.models.common.exporter import TarModelLoader
setup_config()


def main():
    experiments = get_all_experiment_ids()
    print(experiments)
    best_runs = get_best_run()
    print(best_runs)
   #  model = instantiate(config)

    
   #  texts = ["hello"]
   #  encodings = model.backbone.transformation(texts)

   #  output = model(encodings)
   #  print(output.shape) 
   

if __name__ == '__main__':
    main()