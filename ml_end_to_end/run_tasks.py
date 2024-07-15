from ml_end_to_end.config_schemas.config_schema import Config
from ml_end_to_end.utils.config_utils import get_config


@get_config(config_path="../configs", config_name="config")
def run_tasks(config: Config) -> None:
    print(config)


if __name__ == "__main__":
    run_tasks()  # type: ignore
