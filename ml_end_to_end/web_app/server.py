

import os
from ml_end_to_end.models.common.exporter import TarModelLoader
from ml_end_to_end.utils.config_utils import load_config
from hydra.utils import instantiate
from fastapi import FastAPI
from ml_end_to_end.utils.mlflow_utils import get_client

config = load_config(config_path="../configs/automatically_generated",config_name="config")
tokenizer = instantiate(config.tasks.binary_text_classification_task.data_module.transformation)

model_name = "bert_tiny"
model_version =  "1"

mlflow_client = get_client()
mlflow_model = mlflow_client.get_model_version(name=model_name,version=model_version)
model_path = os.path.join(mlflow_model.source,"exported_model.tar.gz")
model = TarModelLoader(model_path).load()
model.eval()

app = FastAPI()

@app.get("/predict_cyberbullying")
def predict_cyberbullying(text:str)->dict[str,int]:
    tokens = tokenizer([text])
    probs = model(tokens)
    classes = (probs>=0.5).item()
    return {"is_cyberbyullying":int(classes)} 


