#!/usr/bin/env bash
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${PRETRAINED_MODEL}" ]; then
  echo "The required environment variable PRETRAINED_MODEL has not been set"
  exit 1
fi

IMAGE_NAME=${IMAGE_NAME:-model-zoo:pytorch-gpu-ssd-mobilenet-inference}
DOCKER_ARGS=${DOCKER_ARGS:---rm -it}

# inference scripts:
# inference_with_dummy_data.sh
# inference_blocked_format.sh
# inference_channels_last.sh
export SCRIPT="${SCRIPT:-inference_with_dummy_data.sh}"

# The dataset directory is not required for the synthetic data script
if [[ ${SCRIPT} != inference_with_dummy_data.sh ]]; then
  if [ -z "${DATASET_DIR}" ]; then
    echo "The required environment variable DATASET_DIR has not been set"
    exit 1
  fi
fi

export label=/workspace/pytorch-gpu-ssd-mobilenet-inference/labels/voc-model-labels.txt

if [[ ${SCRIPT} != quickstart* ]]; then
  SCRIPT="quickstart/$SCRIPT"
fi

VIDEO=$(getent group video | sed -E 's,^video:[^:]*:([^:]*):.*$,\1,')
RENDER=$(getent group render | sed -E 's,^render:[^:]*:([^:]*):.*$,\1,')

test -z "$RENDER" || RENDER_GROUP="--group-add ${RENDER}"

docker run \
  --group-add ${VIDEO} \
  ${RENDER_GROUP} \
  --device=/dev/dri \
  --ipc=host \
  --env DATASET_DIR=${DATASET_DIR} \
  --env PRETRAINED_MODEL=${PRETRAINED_MODEL} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env label=${label} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --env no_proxy=${no_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_ARGS} \
  $IMAGE_NAME \
  /bin/bash $SCRIPT
