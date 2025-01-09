#!/bin/bash -e
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TENSORRTLLM_BACKEND_COMMIT=
GIT_TOKEN=
GIT_REPO=

get_options() {
    while :; do
        case $1 in
        -h | -\? | --help)
            show_help
            exit
            ;;
    --tensorrtllm-backend-commit)
            if [ "$2" ]; then
                TENSORRTLLM_BACKEND_COMMIT=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
    --git-token)
            if [ "$2" ]; then
                GIT_TOKEN=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
    --git-repo)
            if [ "$2" ]; then
                GIT_REPO=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
         -?*)
	    error 'ERROR: Unknown option: ' $1
            ;;
	 ?*)
	    error 'ERROR: Unknown option: ' $1
            ;;
        *)
            break
            ;;
        esac

        shift
    done
}

show_options() {
    echo ""
    echo "Getting TENSORRTLLM Backend Repo"
    echo ""
    echo "   TENSORRTLLM Backend Commit: '${TENSORRTLLM_BACKEND_COMMIT}'"
    echo ""
}


show_help() {
    echo "usage: clone_tensorrtllm.sh"
    echo "  [--tensorrtllm-backend-commit commit]"
    echo "  [--git-token git-token]"
    echo "  [--git-repo git-repo]"
    exit 0
}

missing_requirement() {
    error "ERROR: $1 requires an argument."
}

error() {
    printf '%s %s\n' "$1" "$2" >&2
    exit 1
}

get_options "$@"

if [ -z ${GIT_REPO} ]; then
       GIT_REPO="github.com/triton-inference-server/tensorrtllm_backend"
fi

if [ ! -z ${GIT_TOKEN} ]; then
    GIT_REPO="https://oauth2:${GIT_TOKEN}@${GIT_REPO}"
else
    GIT_REPO="https://${GIT_REPO}"
fi

show_options

git clone ${GIT_REPO}
cd tensorrtllm_backend
git reset --hard ${TENSORRTLLM_BACKEND_COMMIT}
git submodule update --init --recursive
git lfs install
git lfs pull
cd ..
mv tensorrtllm_backend /
