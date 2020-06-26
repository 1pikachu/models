# Copyright (c) 2020 Intel Corporation
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
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG TENSORFLOW_IMAGE="intel/intel-optimized-tensorflow"
ARG TENSORFLOW_TAG

FROM ${TENSORFLOW_IMAGE}:${TENSORFLOW_TAG}


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing python-tk libsm6 libxext6 -y && \
    pip install requests


RUN apt-get update && \
  apt-get install -y software-properties-common
RUN apt-get update && \
  add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
  apt-get install -y gcc-8 g++-8 && \
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 && \
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/v1.18.3/bin/linux/amd64/kubectl && \
  chmod +x ./kubectl && \
  mv kubectl /usr/local/bin

RUN apt-get install openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev -y

RUN apt install -y openssh-server openssh-client && systemctl enable ssh

RUN pip install horovod==0.19.1
