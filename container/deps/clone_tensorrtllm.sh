#!/bin/bash -e
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

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
