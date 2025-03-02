#!/bin/bash -e
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


TAG=
RUN_PREFIX=
PLATFORM=linux/amd64

# Get short commit hash
commit_id=$(git rev-parse --short HEAD)

# if COMMIT_ID matches a TAG use that
current_tag=$(git describe --tags --exact-match 2>/dev/null | sed 's/^v//') || true

# Get latest TAG and add COMMIT_ID for dev
latest_tag=$(git describe --tags --abbrev=0 $(git rev-list --tags --max-count=1 main) | sed 's/^v//') || true
if [[ -z ${latest_tag} ]]; then
    latest_tag="0.0.1"
    echo "No git release tag found, setting to unknown version: ${latest_tag}"
fi

# Use tag if available, otherwise use latest_tag.dev.commit_id
VERSION=v${current_tag:-$latest_tag.dev.$commit_id}

PYTHON_PACKAGE_VERSION=${current_tag:-$latest_tag.dev+$commit_id}

# Frameworks
#
# Each framework has a corresponding base image.  Additional
# dependencies are specified in the /container/deps folder and
# installed within framework specific sections of the Dockerfile.

declare -A FRAMEWORKS=(["STANDARD"]=1 ["TENSORRTLLM"]=2 ["VLLM"]=3 ["VLLM_NIXL"]=4)
DEFAULT_FRAMEWORK=STANDARD

SOURCE_DIR=$(dirname "$(readlink -f "$0")")
DOCKERFILE=${SOURCE_DIR}/Dockerfile
BUILD_CONTEXT=$(dirname "$(readlink -f "$SOURCE_DIR")")

# Base Images

STANDARD_BASE_VERSION=25.01
STANDARD_BASE_IMAGE=nvcr.io/nvidia/tritonserver
STANDARD_BASE_IMAGE_TAG=${STANDARD_BASE_VERSION}-py3

TENSORRTLLM_BASE_VERSION=25.01
TENSORRTLLM_BASE_IMAGE=nvcr.io/nvidia/tritonserver
TENSORRTLLM_BASE_IMAGE_TAG=${TENSORRTLLM_BASE_VERSION}-trtllm-python-py3
# IMPORTANT NOTE: Ensure the repo tag complies with the TRTLLM backend version
# used in the base image above.
TENSORRTLLM_BACKEND_REPO_TAG=triton-llm/v0.17.0
# Set this as 1 to rebuild and replace trtllm backend bits in the container.
# This will allow building triton distributed container image with custom
# trt-llm backend repo branch.
TENSORRTLLM_BACKEND_REBUILD=0
# Set this as 1 to skip cloning the trt-llm backend repo. If cloning is skipped, trt-llm
# backend repo tag and rebuild flag will be ignored. Use this option if you are using
# trtllm llmapi worker.
TENSORRTLLM_SKIP_CLONE=0

VLLM_BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
VLLM_BASE_IMAGE_TAG="25.01-cuda12.8-devel-ubuntu24.04"

VLLM_NIXL_BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base"
VLLM_NIXL_BASE_IMAGE_TAG="25.01-cuda12.8-devel-ubuntu24.04"

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
        --tensorrtllm-backend-repo-tag)
            if [ "$2" ]; then
                TRTLLM_BACKEND_COMMIT=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --tensorrtllm-backend-rebuild)
            if [ "$2" ]; then
                TRTLLM_BACKEND_REBUILD=$2
                shift
            else
                missing_requirement $1
            fi
            ;;
        --skip-clone-tensorrtllm)
            if [ "$2" ]; then
                TENSORRTLLM_SKIP_CLONE=$2
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
        --target)
            if [ "$2" ]; then
                TARGET=$2
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
                TAG="--tag $2"
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
        --plain-progress)
            PLAIN_PROGRESS=" --progress=plain"
            ;;
        --cache-from)
            if [ "$2" ]; then
                CACHE_FROM="--cache-from $2"
                shift
            else
                missing_requirement $1
            fi
            ;;
        --cache-to)
            if [ "$2" ]; then
                CACHE_TO="--cache-to $2"
                shift
            else
                missing_requirement $1
            fi
            ;;
        --build-context)
            if [ "$2" ]; then
                BUILD_CONTEXT_ARG="--build-context $2"
                shift
            else
                missing_requirement $1
            fi
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
        TAG="--tag triton-distributed:${VERSION}-${FRAMEWORK,,}"
        if [ ! -z ${TARGET} ]; then
            TAG="${TAG}-${TARGET}"
        fi
    fi

    if [ ! -z "$PLATFORM" ]; then
        PLATFORM="--platform ${PLATFORM}"
    fi

    if [ ! -z "$TARGET" ]; then
        TARGET_STR="--target ${TARGET}"
    fi
}


show_image_options() {
    echo ""
    echo "Building Triton Distributed Image: '${TAG}'"
    echo ""
    echo "   Base: '${BASE_IMAGE}'"
    echo "   Base_Image_Tag: '${BASE_IMAGE_TAG}'"
    if [[ $FRAMEWORK == "TENSORRTLLM" ]]; then
        echo "   Tensorrtllm Backend Repo Tag: '${TENSORRTLLM_BACKEND_REPO_TAG}'"
        echo "   Tensorrtllm Backend Rebuild: '${TENSORRTLLM_BACKEND_REBUILD}'"
        echo "   Tensorrtllm Skip Clone: '${TENSORRTLLM_SKIP_CLONE}'"
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
    echo "  [--tensorrtllm-backend-repo-tag commit or tag]"
    echo "  [--tensorrtllm-backend-rebuild whether or not to rebuild the backend]"
    echo "  [--skip-clone-tensorrtllm whether or not to skip cloning the trt-llm backend repo]"
    echo "  [--build-arg additional build args to pass to docker build]"
    echo "  [--cache-from cache location to start from]"
    echo "  [--cache-to location where to cache the build output]"
    echo "  [--tag tag for image]"
    echo "  [--no-cache disable docker build cache]"
    echo "  [--plain-progress print docker outputs without progress bar]"
    echo "  [--dry-run print docker commands without running]"
    echo "  [--build-context name=path to add build context]"
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


# Update DOCKERFILE if framework is VLLM
if [[ $FRAMEWORK == "VLLM" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.vllm
elif [[ $FRAMEWORK == "VLLM_NIXL" ]]; then
    DOCKERFILE=${SOURCE_DIR}/Dockerfile.vllm_nixl
fi

# BUILD DEV IMAGE

BUILD_ARGS+=" --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG --build-arg FRAMEWORK=$FRAMEWORK --build-arg ${FRAMEWORK}_FRAMEWORK=1 --build-arg VERSION=$VERSION --build-arg PYTHON_PACKAGE_VERSION=$PYTHON_PACKAGE_VERSION"

if [ ! -z ${GITHUB_TOKEN} ]; then
    BUILD_ARGS+=" --build-arg GITHUB_TOKEN=${GITHUB_TOKEN} "
fi

if [ ! -z ${GITLAB_TOKEN} ]; then
    BUILD_ARGS+=" --build-arg GITLAB_TOKEN=${GITLAB_TOKEN} "
fi

if [[ $FRAMEWORK == "TENSORRTLLM" ]] && [ ! -z ${TENSORRTLLM_BACKEND_REPO_TAG} ]; then
    BUILD_ARGS+=" --build-arg TENSORRTLLM_BACKEND_REPO_TAG=${TENSORRTLLM_BACKEND_REPO_TAG} "
    BUILD_ARGS+=" --build-arg TENSORRTLLM_BACKEND_REBUILD=${TENSORRTLLM_BACKEND_REBUILD} "
    BUILD_ARGS+=" --build-arg TENSORRTLLM_SKIP_CLONE=${TENSORRTLLM_SKIP_CLONE} "
fi

if [ ! -z ${HF_TOKEN} ]; then
    BUILD_ARGS+=" --build-arg HF_TOKEN=${HF_TOKEN} "
fi

LATEST_TAG="--tag triton-distributed:latest-${FRAMEWORK,,}"
if [ ! -z ${TARGET} ]; then
    LATEST_TAG="${LATEST_TAG}-${TARGET}"
fi

show_image_options

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

$RUN_PREFIX docker buildx build -f $DOCKERFILE $TARGET_STR $PLATFORM $BUILD_ARGS $CACHE_FROM $CACHE_TO --output type=docker $TAG $LATEST_TAG $BUILD_CONTEXT_ARG $BUILD_CONTEXT $NO_CACHE $PLAIN_PROGRESS

{ set +x; } 2>/dev/null

if [ -z "$RUN_PREFIX" ]; then
    set -x
fi

{ set +x; } 2>/dev/null
