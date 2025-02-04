# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE="ubuntu:noble"

FROM ${BASE_IMAGE}

# Set useful labels.
LABEL "base"="ubuntu:noble"
LABEL "configuration"="tester"
LABEL "version"="devel"

# Stop APT (Debian package manager) from complaining about interactivity.
ENV DEBIAN_FRONTEND=noninteractive

# Set additional environment values that make usage more pleasant.
ENV TERM=xterm-256color

# Update / upgrade the base image.
RUN apt-get update \
 && apt-get upgrade --yes \
 && rm -rf /var/lib/apt/lists/*

# Install pre-requisites.
RUN apt-get update \
 && apt-get install --no-install-recommends --yes --fix-missing \
    apt-transport-https \
    software-properties-common \
    git \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Download and register the Microsoft repository keys.
RUN wget -q https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb \
 && dpkg -i packages-microsoft-prod.deb \
 && rm packages-microsoft-prod.deb

# Install Powershell runtime and terminal.
RUN apt-get update \
 && apt-get install --yes \
    powershell

# Download Helm and move it to /usr/local/bin.
RUN wget -q https://get.helm.sh/helm-v3.17.0-linux-amd64.tar.gz \
 && tar -zxvf helm-v3.17.0-linux-amd64.tar.gz \
 && mv linux-amd64/helm /usr/local/bin/helm

# Create /workspace and set it to the default folder.
RUN mkdir -p /workspace

# Enable Git operations in the /workspace directory.
RUN printf "[safe]\n      directory=/workspace\n" > /root/.gitconfig

WORKDIR /workspace
