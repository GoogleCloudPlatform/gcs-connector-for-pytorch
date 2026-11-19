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

    echo Installing required dependencies.
    pip install --no-cache-dir .

    echo Installing checkpointing requirements
    pip install --no-cache-dir -r ./dataflux_pytorch/benchmark/requirements.txt
}

function run_unit_tests() {
    echo Running unit tests.
    python3 -m pytest dataflux_pytorch/tests -vv --junit-xml="${KOKORO_ARTIFACTS_DIR}/unit_tests/sponge_log.xml" --log-cli-level=DEBUG
}

function run_integration_tests(){
    echo Running basic integration test.
    python3 -m pytest dataflux_pytorch/integration_tests/integration_test.py -vv --junit-xml="${KOKORO_ARTIFACTS_DIR}/integration_tests/sponge_log.xml" --log-cli-level=DEBUG
    echo Running checkpoint integration test.
    python3 dataflux_pytorch/benchmark/checkpointing/singlenode/train.py --project=dataflux-project --ckpt-dir-path=gs://df-ckpt-presubmit/ --layers=10 --steps=5
}

setup_virtual_envs
run_git_commands
install_requirements
run_unit_tests
run_integration_tests
