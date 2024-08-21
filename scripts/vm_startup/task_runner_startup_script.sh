#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

export GCP_LOGGING_ENABLED="TRUE"

# Obtención de atributos de la instancia desde la metadata
INSTANCE_GROUP_NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance_group_name -H "Metadata-Flavor: Google")
ZONE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/zone -H "Metadata-Flavor: Google")
PYTHON_HASH_SEED=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/python_hash_seed -H "Metadata-Flavor: Google" || echo "42")
NODE_COUNT=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/node_count -H "Metadata-Flavor: Google")
DISKS=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/disks -H "Metadata-Flavor: Google")
ETCD_IP=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/etcd_ip -H "Metadata-Flavor: Google")



# Convertir el nombre del grupo de instancias a minúsculas
INSTANCE_GROUP_NAME=$(echo ${INSTANCE_GROUP_NAME} | tr '[:upper:]' '[:lower:]')

echo -e " TRAINING: instance group name: ${INSTANCE_GROUP_NAME}, docker_image: ${DOCKER_IMAGE}, node_count: ${NODE_COUNT}, python_hash_seed: ${PYTHON_HASH_SEED}"

echo "=========== Installing Nvidia Drivers ==========="
apt-get update && /opt/deeplearning/install-driver.sh
echo "=========== Downloading docker image ==========="
gcloud auth configure-docker --quiet europe-west1-docker.pkg.dev
time docker pull "${DOCKER_IMAGE}"

echo "=========== Training start ==========="
if ["${ETCD_IP}" = "None"]; then
docker run --init --rm --gpus all --ipc host --user root  --hostname "$(hostname)" --privileged --log-driver=gcplogs \
 -e PYTHONHASHSEED="${PYTHON_HASH_SEED}" \
 -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
 -e TOKENIZERS_PARALLELISM=false \
 ${DOCKER_IMAGE} \
 torchrun  \
 --nnodes="${NODE_COUNT}"\
 --nproc_per_node='gpu'\
 ml_end_to_end/run_tasks.py || echo "=========== Training: job failed ==========="
else
docker run --init --rm --gpus all --ipc host --user root  --hostname "$(hostname)" --privileged --log-driver=gcplogs \
 -e PYTHONHASHSEED="${PYTHON_HASH_SEED}" \
 -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
 -e TOKENIZERS_PARALLELISM=false \
 ${DOCKER_IMAGE} \
 torchrun  \
 --nnodes="${NODE_COUNT}"\
 --nproc_per_node='gpu'\
 --rdzv_id="${INSTANCE_GROUP_NAME}$"\
 --rdzv_backend=etcd-v2\
 --rdzv_endpoint="${ETCD_IP}"\
 ml_end_to_end/run_tasks.py || echo "=========== Training: job failed ==========="
fi

echo "=========== Cleaning up ==========="
# Intento de eliminar el grupo de instancias
echo "Deleting instance group ${INSTANCE_GROUP_NAME}"
gcloud compute instance-groups managed delete --quiet "${INSTANCE_GROUP_NAME}" --zone "${ZONE}"
