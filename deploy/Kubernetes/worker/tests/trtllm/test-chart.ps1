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

set-strictmode -version latest

. "$(& git rev-parse --show-toplevel)/deploy/Kubernetes/_build/helm-test.ps1"

$is_debug = $false

for ($i = 0 ; $i -lt $args.count ; $i += 1) {
  $arg = $args[$i]
  if ('--debug' -ieq $arg) {
    $is_debug = $true
  }
  elseif (('--verbose' -ieq $arg) -or ('-v' -ieq $arg)) {
    set_verbosity('DETAILED')
  }
  else {
    fatal-exit "Unknown option '${arg}'."
  }
}

$is_debug = $is_debug -or $(is_debug)
if ($is_debug) {
  $DebugPreference = 'Continue'
}
else {
  $DebugPreference = 'SilentlyContinue'
}

$tests = @(
    @{
      name = 'worker/trtllm/basic'
      expected = 0
      matches = @(
          'helm.sh/chart: "triton-distributed_worker-trtllm"[\n\r]{1,2}'
          'app.kubernetes.io/instance: test[\n\r]{1,2}'
          'app.kubernetes.io/name: faux-triton[\n\r]{1,2}'
          '- TRITON_MODEL_REPOSITORY: "/var/run/models"[\n\r]{1,2}'
          '- TRITON_MODEL_GENERATION_PATH: "/var/run/trtllm"[\n\r]{1,2}'
          'image: some_false-container_name:with_a-tag[\n\r]{1,2}'
          'ephemeral-storage: 98Gi[\n\r]{1,2}'
        )
      options = @()
      values = @('basic_values.yaml')
    }
    @{
      name = 'worker/trtllm/basic_error'
      expected = 1
      matches = @(
          '- triton: componentName is required[\n\r]{1,2}'
          'image: name is required'
        )
      options = @()
      values = @()
    }
    @{
      name = 'worker/trtllm/volume_mounts'
      expected = 0
      matches = @(
          '- name: mount_w_path[\n\r ]+persistentVolumeClaim:[\n\r ]+claimName: w_path_pvc[\n\r]{1,2}'
          '- name: mount_wo_path[\n\r ]+persistentVolumeClaim:[\n\r ]+claimName: wo_path_pvc[\n\r]{1,2}'
          '- mountPath: \/var\/run\/models\/subpath[\n\r ]+name: mount_w_path[\r\n]{1,2}'
          '- mountPath: \/var\/run\/models[\n\r ]+name: mount_wo_path[\n\r]{1,2}'
        )
      options = @()
      values = @(
          'basic_values.yaml'
          'volume_mounts.yaml'
        )
    }
    @{
      name = 'worker/trtllm/bad_volume_mounts'
      expected = 1
      matches = @(
          '- modelRepository.volumeMounts.0: persistentVolumeClaim is required'
        )
      options = @()
      values = @(
          'basic_values.yaml'
          'bad_volume_mounts.yaml'
        )
    }
    @{
      name = 'worker/trtllm/host_cache'
      expected = 0
      matches = @(
          'ephemeral-storage: 2Gi[\n\r]{1,2}'
          '- name: model-host-cache[\n\r ]+hostPath:[\n\r ]+path: /triton/trtllm-cache[\n\r ]+type: DirectoryOrCreate[\n\r]{1,2}'
          '- TRITON_MODEL_GENERATION_QUOTA: "200Gi"[\n\r]{1,2}'
        )
      options = @()
      values = @(
          'basic_values.yaml'
          'host_cache.yaml'
        )
    }
    @{
      name = 'worker/trtllm/host_cache'
      expected = 0
      matches = @(
          'ephemeral-storage: 202Gi[\n\r]{1,2}'
          '- name: model-host-cache[\n\r ]+emptyDir:[\n\r ]+sizeLimit: 200Gi[\n\r]{1,2}'
          '- TRITON_MODEL_GENERATION_QUOTA: "200Gi"[\n\r]{1,2}'
        )
      options = @()
      values = @(
          'basic_values.yaml'
          'non_host_cache.yaml'
        )
    }
  )

try {
  test_helm_chart 'deploy/Kubernetes/worker/charts/trtllm' 'deploy/Kubernetes/worker/tests/trtllm' $tests
}
catch {
  if ($is_debug) {
    throw $_
  }

  fatal_exit "$_"
}

# Clean up any NVBUILD environment variables left behind by the build.
cleanup_after
