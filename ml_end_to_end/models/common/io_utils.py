import os

from typing import Any, Dict

import yaml

from fsspec import AbstractFileSystem, filesystem

GCS_PREFIX = "gs://"
GCS_FILE_SYSTEM_NAME = "gcs"
LOCAL_FILE_SYSTEM_NAME = "file"
TMP_FILE_PATH = "/tmp/"


def choose_file_file(path: str) -> AbstractFileSystem:
    return filesystem(GCS_FILE_SYSTEM_NAME) if path.startswith(GCS_PREFIX) else filesystem(LOCAL_FILE_SYSTEM_NAME)


def open_file(path: str, mode: str = "r") -> Any:
    file_system = choose_file_file(path)
    return file_system.open(path, mode)


def write_yaml_file(yaml_file_path: str, yaml_file_content: Dict[Any, Any]) -> None:
    with open_file(yaml_file_path, "w") as f:
        yaml.dump(yaml_file_content, f)


def is_dir(path: str) -> bool:
    file_system = choose_file_file(path)
    isdir: bool = file_system.isdir(path)
    return isdir


def is_file(path: str) -> bool:
    file_system = choose_file_file(path)
    isfile: bool = file_system.isfile(path)
    return isfile


def make_dirs(path: str) -> None:
    file_system = choose_file_file(path)
    file_system.makedirs(path, exist_ok=True)


def list_paths(path: str) -> list[str]:
    file_system = choose_file_file(path)
    if not is_dir(path):
        return []

    paths: list[str] = file_system.ls(path)
    if GCS_FILE_SYSTEM_NAME in file_system.protocol:
        gs_paths: list[str] = [f"{GCS_PREFIX}{path}" for path in paths]
        return gs_paths
    return paths


def copy_dir(source: str, destination: str) -> None:
    if not is_dir(destination):
        make_dirs(destination)
    source_files = list_paths(source)
    for source_file in source_files:
        target_file = os.path.join(destination, os.path.basename(source_file))
        if is_file(source_file):
            with open_file(source_file, mode="rb") as source_f, open_file(target_file, mode="wb") as target:
                content = source_f.read()
                target.write(content)
        else:
            raise ValueError(f"Source file {source_file} is not vlaid")


def translate_gcs_dir_to_local(path: str) -> str:
    if path.startswith(GCS_PREFIX):
        path = path.rstrip("/")
        local_path = os.path.join(TMP_FILE_PATH,os.path.split(path)[-1])
        os.makedirs(local_path, exist_ok=True)
        copy_dir(path,local_path)
        return local_path
    
    return path


def convert_gcs_path_to_local_path(path: str) -> str:
    if path.startswith(GCS_PREFIX):
        path = path.rstrip("/")
        local_path = os.path.join(TMP_FILE_PATH,os.path.split(path)[-1])
        return local_path
    
    return path

def copy_file(source_file:str,target_path:str)->None:
    with open_file(source_file,mode="rb") as source,open_file(target_path,"wb") as target:
        content = source.read()
        target.write(content)




def cache_gcs_resource_locally(path: str) -> str:
    if path.startswith(GCS_PREFIX):
        local_path = convert_gcs_path_to_local_path()

        if os.path.exists(local_path):
            return local_path
    
        if is_dir(path):
            os.makedirs(local_path,exist_ok=True)
            copy_dir(path, local_path)
        else:
            copy_file(path,local_path)

        return local_path
    
    return path
