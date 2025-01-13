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

# A script to download a Python wheel, patch it, copy additional files,
# repackage it, and optionally install the new wheel.
#
# Usage examples:
#   ./script.sh --debug
#   ./script.sh --install --force
#   ./script.sh --help

###############################################################################
#  CONFIGURATION & DEFAULTS
###############################################################################
DEFAULT_WHEEL_URL="https://files.pythonhosted.org/packages/4a/4c/ee65ba33467a4c0de350ce29fbae39b9d0e7fcd887cc756fa993654d1228/vllm-0.6.3.post1-cp38-abi3-manylinux1_x86_64.whl"
DEFAULT_PATCH_FILE="vllm_patch_063post1.patch"
DEFAULT_DATA_PLANE_DIR="data_plane"
DEFAULT_WHEEL_DIR="wheel"
DEFAULT_OUTPUT_WHEEL="vllm-dist-0.6.3.post1-cp38-abi3-manylinux1_x86_64.whl"
# Optionally set a default SHA256 checksum for the downloaded wheel
# DEFAULT_CHECKSUM="SOME_SHA256_HERE"

###############################################################################
#  HELPER FUNCTIONS
###############################################################################
usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Options:
  -u, --url <URL>              Wheel URL to download (default: $DEFAULT_WHEEL_URL)
  -p, --patch <FILE>           Patch file path (default: $DEFAULT_PATCH_FILE)
  -d, --data-plane <DIR>       Directory with additional data-plane files (default: $DEFAULT_DATA_PLANE_DIR)
  -w, --wheel-dir <DIR>        Extract destination directory (default: $DEFAULT_WHEEL_DIR)
  -o, --output-wheel <FILE>    Name/path for the repackaged wheel (default: $DEFAULT_OUTPUT_WHEEL)
  -f, --force                  Force overwriting existing directories/files without prompts
  -i, --install                Install the new wheel after repackaging
  -D, --debug                  Enable debug mode (verbose output)
  -h, --help                   Show this help and exit

Example:
  $0 \\
    --url "https://example.com/path/to/vllm.whl" \\
    --patch my_patch.patch \\
    --data-plane custom_data/ \\
    --wheel-dir extracted_wheel \\
    --output-wheel my_vllm_dist.whl \\
    --install \\
    --force \\
    --debug

EOF
}

info_log() {
  # Always prints, showing a timestamp.
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $*"
}

debug_log() {
  # Prints only if DEBUG=true
  if [[ "$DEBUG" == true ]]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] DEBUG: $*"
  fi
}

error_exit() {
  echo "ERROR: $*" >&2
  exit 1
}

###############################################################################
#  PARSE ARGUMENTS
###############################################################################
FORCE_OVERWRITE=false
INSTALL_WHEEL=false
DEBUG=false

WHEEL_URL="$DEFAULT_WHEEL_URL"
PATCH_FILE="$DEFAULT_PATCH_FILE"
DATA_PLANE_DIR="$DEFAULT_DATA_PLANE_DIR"
WHEEL_DIR="$DEFAULT_WHEEL_DIR"
OUTPUT_WHEEL="$DEFAULT_OUTPUT_WHEEL"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -u|--url)
      WHEEL_URL="$2"
      shift 2
      ;;
    -p|--patch)
      PATCH_FILE="$2"
      shift 2
      ;;
    -d|--data-plane)
      DATA_PLANE_DIR="$2"
      shift 2
      ;;
    -w|--wheel-dir)
      WHEEL_DIR="$2"
      shift 2
      ;;
    -o|--output-wheel)
      OUTPUT_WHEEL="$2"
      shift 2
      ;;
    -f|--force)
      FORCE_OVERWRITE=true
      shift
      ;;
    -i|--install)
      INSTALL_WHEEL=true
      shift
      ;;
    -D|--debug)
      DEBUG=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      error_exit "Unknown option: $1"
      ;;
  esac
done

###############################################################################
#  MAIN SCRIPT
###############################################################################
# Enable debug mode if requested
if [[ "$DEBUG" == true ]]; then
  set -x
fi

info_log "Starting wheel patching script..."

# ---------------------------------------------------------------------------
# 1. Check for existing wheel file or directory
# ---------------------------------------------------------------------------
WHEEL_FILENAME=$(basename "$WHEEL_URL")

if [[ -f "$WHEEL_FILENAME" && "$FORCE_OVERWRITE" != true ]]; then
  info_log "File '$WHEEL_FILENAME' already exists. Reusing existing file."
  info_log "If you want to redownload, remove '$WHEEL_FILENAME' or use --force."
else
  info_log "Downloading wheel from $WHEEL_URL..."
  rm -f "$WHEEL_FILENAME" 2>/dev/null || true  # Remove existing file if forcing
  wget -O "$WHEEL_FILENAME" "$WHEEL_URL"
fi

# ---------------------------------------------------------------------------
# 2. Optional: Verify checksum (commented out by default)
# ---------------------------------------------------------------------------
# if [[ -n "$DEFAULT_CHECKSUM" ]]; then
#   info_log "Verifying SHA256 checksum..."
#   echo "${DEFAULT_CHECKSUM}  ${WHEEL_FILENAME}" | sha256sum --check - || error_exit "Checksum mismatch!"
# fi

# ---------------------------------------------------------------------------
# 3. Create/clean wheel extraction directory
# ---------------------------------------------------------------------------
if [[ -d "$WHEEL_DIR" ]]; then
  if [[ "$FORCE_OVERWRITE" == true ]]; then
    info_log "Removing existing directory '$WHEEL_DIR' due to --force..."
    rm -rf "$WHEEL_DIR"
  else
    error_exit "Directory '$WHEEL_DIR' already exists. Use --force to overwrite."
  fi
fi

info_log "Creating directory '$WHEEL_DIR'..."
mkdir -p "$WHEEL_DIR"

# ---------------------------------------------------------------------------
# 4. Unzip the wheel into the specified directory
# ---------------------------------------------------------------------------
info_log "Unzipping wheel into directory '$WHEEL_DIR'..."
unzip -q "$WHEEL_FILENAME" -d "$WHEEL_DIR"

# ---------------------------------------------------------------------------
# 5. Check/Apply patch
# ---------------------------------------------------------------------------
if [[ ! -f "$PATCH_FILE" ]]; then
  error_exit "Patch file '$PATCH_FILE' not found."
fi

PATCH_TARGET_DIR="$WHEEL_DIR/vllm"
if [[ ! -d "$PATCH_TARGET_DIR" ]]; then
  error_exit "Could not find directory '$PATCH_TARGET_DIR' in unzipped wheel."
fi

info_log "Applying patch '$PATCH_FILE' to '$PATCH_TARGET_DIR'..."
debug_log "Executing: (cd \"$PATCH_TARGET_DIR\" && patch -p1 < \"../../$PATCH_FILE\")"
(
  cd "$PATCH_TARGET_DIR"
  patch -p1 < "../../$PATCH_FILE"
)

# ---------------------------------------------------------------------------
# 6. Copy data plane files
# ---------------------------------------------------------------------------
if [[ ! -d "$DATA_PLANE_DIR" ]]; then
  error_exit "Data plane directory '$DATA_PLANE_DIR' not found."
fi

info_log "Copying files from '$DATA_PLANE_DIR' to '$PATCH_TARGET_DIR/vllm/distributed'..."
mkdir -p "$PATCH_TARGET_DIR/vllm/distributed"
cp -r "$DATA_PLANE_DIR/"* "$PATCH_TARGET_DIR/vllm/distributed/"

# ---------------------------------------------------------------------------
# 7. Re-package into a new wheel
# ---------------------------------------------------------------------------
if [[ -f "$OUTPUT_WHEEL" && "$FORCE_OVERWRITE" != true ]]; then
  error_exit "Output wheel '$OUTPUT_WHEEL' already exists. Use --force to overwrite."
fi

info_log "Creating new wheel file '$OUTPUT_WHEEL'..."
debug_log "Executing: (cd \"$WHEEL_DIR\" && zip -rq \"../$OUTPUT_WHEEL\" .)"
(
  cd "$WHEEL_DIR"
  zip -rq "../$OUTPUT_WHEEL" .
)

# ---------------------------------------------------------------------------
# 8. Optional: Install the new wheel
# ---------------------------------------------------------------------------
if [[ "$INSTALL_WHEEL" == true ]]; then
  # Check if pip is installed
  if ! command -v pip >/dev/null 2>&1; then
    error_exit "pip is not installed or not found in PATH."
  fi

  info_log "Installing newly created wheel '$OUTPUT_WHEEL'..."
  pip install --force-reinstall --upgrade "$OUTPUT_WHEEL"
fi

info_log "Patch and repackage completed successfully!"
info_log "New wheel: $OUTPUT_WHEEL"
if [[ "$INSTALL_WHEEL" == true ]]; then
  info_log "Wheel has been installed."
fi

