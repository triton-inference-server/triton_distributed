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

TAG=
RUN_PREFIX=
PLATFORM=linux/amd64

# Frameworks
#
# Each framework has a corresponding base image.  Additional
# dependencies are specified in the /container/deps folder and
# installed within framework specific sections of the Dockerfile.

declare -A FRAMEWORKS=(["STANDARD"]=1 ["TENSORRTLLM"]=2 ["VLLM"]=3)
DEFAULT_FRAMEWORK=STANDARD

SOURCE_DIR=$(dirname "$(readlink -f "$0")")
DOCKERFILE=${SOURCE_DIR}/Dockerfile
BUILD_CONTEXT=$(dirname "$(readlink -f "$SOURCE_DIR")")

# Base Images

STANDARD_BASE_VERSION=24.12
STANDARD_BASE_IMAGE=nvcr.io/nvidia/tritonserver
STANDARD_BASE_IMAGE_TAG=${STANDARD_BASE_VERSION}-py3

TENSORRTLLM_BASE_VERSION=24.12
TENSORRTLLM_BASE_IMAGE=nvcr.io/nvidia/tritonserver
TENSORRTLLM_BASE_IMAGE_TAG=${TENSORRTLLM_BASE_VERSION}-trtllm-python-py3
# IMPORTANT NOTE: Ensure the commit matches the TRTLLM backend version used in the base image above
TENSORRTLLM_BACKEND_COMMIT=v0.16.0

VLLM_BASE_VERSION=24.12
VLLM_BASE_IMAGE=nvcr.io/nvidia/tritonserver
VLLM_BASE_IMAGE_TAG=${VLLM_BASE_VERSION}-vllm-python-py3

get_options() {
    while :; do
        case $1 in
        -h | -\? | --help)
            show_help
            exit
            ;;
	--platform)
            if [ "$2" ]; then
                PLATFORM=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
	--framework)
            if [ "$2" ]; then
                FRAMEWORK=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
        --tensorrtllm-backend-commit)
            if [ "$2" ]; then
                TRTLLM_BACKEND_COMMIT=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
        --base-image)
            if [ "$2" ]; then
                BASE_IMAGE=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
	--base-image-tag)
            if [ "$2" ]; then
                BASE_IMAGE_TAG=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
        --build-arg)
            if [ "$2" ]; then
                BUILD_ARGS+="--build-arg $2 "
                shift
            else
		missing_requirement $1
            fi
            ;;
        --tag)
            if [ "$2" ]; then
                TAG=$2
                shift
            else
		missing_requirement $1
            fi
            ;;
        --dry-run)
            RUN_PREFIX="echo"
            echo ""
            echo "=============================="
            echo "DRY RUN: COMMANDS PRINTED ONLY"
            echo "=============================="
            echo ""
            ;;
	--no-cache)
	    NO_CACHE=" --no-cache"
            ;;
        --)
            shift
            break
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

    if [ -z "$FRAMEWORK" ]; then
	FRAMEWORK=$DEFAULT_FRAMEWORK
    fi

    if [ ! -z "$FRAMEWORK" ]; then
	FRAMEWORK=${FRAMEWORK^^}

	if [[ ! -n "${FRAMEWORKS[$FRAMEWORK]}" ]]; then
	    error 'ERROR: Unknown framework: ' $FRAMEWORK
	fi

	if [ -z $BASE_IMAGE_TAG ]; then
	    BASE_IMAGE_TAG=${FRAMEWORK}_BASE_IMAGE_TAG
	    BASE_IMAGE_TAG=${!BASE_IMAGE_TAG}
	fi

	if [ -z $BASE_IMAGE ]; then
	    BASE_IMAGE=${FRAMEWORK}_BASE_IMAGE
	    BASE_IMAGE=${!BASE_IMAGE}
	fi

	if [ -z $BASE_IMAGE ]; then
	    error "ERROR: Framework $FRAMEWORK without BASE_IMAGE"
	fi

	BASE_VERSION=${FRAMEWORK}_BASE_VERSION
	BASE_VERSION=${!BASE_VERSION}

    fi

    if [ -z "$TAG" ]; then
        TAG="triton-distributed:${FRAMEWORK,,}-${BASE_VERSION}"
    fi

    if [ ! -z "$PLATFORM" ]; then
        PLATFORM="--platform ${PLATFORM}"
    fi


}


show_image_options() {
    echo ""
    echo "Building Triton Distributed Image: '${TAG}'"
    echo ""
    echo "   Base: '${BASE_IMAGE}'"
    echo "   Base_Image_Tag: '${BASE_IMAGE_TAG}'"
    if [[ $FRAMEWORK == "TENSORRTLLM" ]]; then
	echo "   Tensorrtllm Backend Commit: '${TENSORRTLLM_BACKEND_COMMIT}'"
    fi
    echo "   Build Context: '${BUILD_CONTEXT}'"
    echo "   Build Arguments: '${BUILD_ARGS}'"
    echo "   Framework: '${FRAMEWORK}'"
    echo ""
}

show_help() {
    echo "usage: build.sh"
    echo "  [--base base image]"
    echo "  [--base-imge-tag base image tag]"
    echo "  [--platform platform for docker build"
    echo "  [--framework framework one of ${!FRAMEWORKS[@]}]"
    echo "  [--tensorrtllm-backend-commit commit or tag]"
    echo "  [--build-arg additional build args to pass to docker build]"
    echo "  [--tag tag for image]"
    echo "  [--no-cache disable docker build cache]"
    echo "  [--dry-run print docker commands without running]"
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


# BUILD DEV IMAGE

BUILD_ARGS+=" --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG --build-arg FRAMEWORK=$FRAMEWORK --build-arg ${FRAMEWORK}_FRAMEWORK=1"

if [ ! -z ${GITHUB_TOKEN} ]; then
    BUILD_ARGS+=" --build-arg GITHUB_TOKEN=${GITHUB_TOKEN} "
fi

if [ ! -z ${GITLAB_TOKEN} ]; then
    BUILD_ARGS+=" --build-arg GITLAB_TOKEN=${GITLAB_TOKEN} "
fi

if [[ $FRAMEWORK == "TENSORRTLLM" ]] && [ ! -z ${TENSORRTLLM_BACKEND_COMMIT} ]; then
    BUILD_ARGS+=" --build-arg TENSORRTLLM_BACKEND_COMMIT=${TENSORRTLLM_BACKEND_COMMIT} "
fi

if [ ! -z ${HF_TOKEN} ]; then
    BUILD_ARGS+=" --build-arg HF_TOKEN=${HF_TOKEN} "
fi

show_image_options


if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

$RUN_PREFIX docker build -f $DOCKERFILE $PLATFORM $BUILD_ARGS -t $TAG $BUILD_CONTEXT $NO_CACHE

{ set +x; } 2>/dev/null

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

{ set +x; } 2>/dev/null
