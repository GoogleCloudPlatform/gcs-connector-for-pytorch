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
    python3 -m venv venv
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
    echo Creating fake application default credentials. 
cat > $HOME/.config/gcloud/application_default_credentials.json << EOF
{
"account": "",
"client_id": "",
"client_secret": "",
"quota_project_id": "",
"refresh_token": "",
"type": "authorized_user",
"universe_domain": "googleapis.com"
}
EOF

    echo Installing requirements.
    pip install -r requirements.txt

    echo Installing required dependencies.
    pip install .
}

function run_unit_tests() {
    echo "Running unit tests on MacOS (x86)."
    python3 -m pytest dataflux_pytorch/tests -vv --junit-xml="${KOKORO_ARTIFACTS_DIR}/unit_tests/sponge_log.xml" --log-cli-level=DEBUG
}

setup_virtual_envs
run_git_commands
install_requirements
run_unit_tests
