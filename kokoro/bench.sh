#!/bin/bash
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fail on any error.
set -e

# Code under repo is checked out to this directory.
cd "${KOKORO_ARTIFACTS_DIR}/github/dataflux-pytorch"

function setup_virtual_envs() {
    echo Setting up Python virtual environment.
    if ! python3 -m venv venv 2>/dev/null; then
        rm -rf venv
        if python3 -m virtualenv --version >/dev/null 2>&1; then
            python3 -m virtualenv venv
        else
            sudo apt-get -y update
            sudo apt install -y python3.8-venv
            python3 -m venv venv
        fi
    fi
    source venv/bin/activate
}

function run_git_commands() {
    echo Setting git permissions.
    git config --global --add safe.directory "*" 

    echo Installing git submodules.
    git submodule init

    echo Updating git submodules.
    git submodule update
}

function install_requirements() {
    if ! command -v pip >/dev/null 2>&1; then
        echo Installing python3-pip.
        sudo apt-get -y update
        sudo apt-get -y install python3-pip
    fi

    echo Installing requirements.
    pip install --no-cache-dir -r requirements.txt

    echo Installing parquet demo requirements.
    pip install --no-cache-dir -r ./demo/lightning/text_based/distributed/requirements.txt

    echo Installing image training demo requirements.
    pip install --no-cache-dir -r ./demo/lightning/image_segmentation/requirements.txt

    echo Installing required dependencies.
    pip install --no-cache-dir .
}

function run_benchmarks(){
    echo Running benchmarks...
    echo Running parquet-text benchmark.
    python3 -u ./demo/lightning/text_based/distributed/model.py --local --project=dataflux-project --bucket=fineweb-df-benchmark --num-workers=2 --num-nodes=1 --devices=5 --batch-size=512 --epochs=5 --limit-train-batches=1000 --log-level=ERROR;
    echo Running image-segmentation benchmark.
    python3 -u ./demo/lightning/image_segmentation/train.py --local --benchmark --gcp_project=dataflux-project --gcs_bucket=dataflux-demo-public --images_prefix=image-segmentation-dataset/images --labels_prefix=image-segmentation-dataset/labels --num_nodes=1 --num_devices=5 --epochs=2;
    echo Installing checkpointing demo requirements.
    pip install --no-cache-dir -r ./dataflux_pytorch/benchmark/requirements.txt
    echo Running single node checkpointing benchmark.
    python3 -u ./dataflux_pytorch/benchmark/checkpointing/singlenode/train.py --project=dataflux-project --ckpt-dir-path=gs://df-ckpt-presubmit/ --layers=1000 --steps=5
    echo Running single node async checkpointing benchmark.
    python3 -u ./dataflux_pytorch/benchmark/checkpointing/singlenode/train.py --project=dataflux-project --ckpt-dir-path=gs://df-ckpt-presubmit/async/ --layers=1000 --steps=5 --checkpoint=async_df_lightning
    echo Running multinode simulated checkpointing benchmark.
    python3 -u demo/lightning/checkpoint/simulated/multiprocessing_train.py --project=dataflux-project --ckpt-dir-path=gs://df-ckpt-presubmit/multinode-simulated/ --layer-size=1000 --padding-size=1000 --world-size=4
}

setup_virtual_envs
run_git_commands
install_requirements
run_benchmarks
