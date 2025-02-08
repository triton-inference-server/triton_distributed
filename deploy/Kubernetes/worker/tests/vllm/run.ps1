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

if ($null -eq $(get-command 'git' -ea 0)) {
  throw "Required tool 'git' not found, unable to continue."
}
. "$(& git rev-parse --show-toplevel)/deploy/Kubernetes/_build//helm-test.ps1"

$componentName = 'worker'
$componentType = 'vllm'

$tests = @(
    @{
      name = 'basic'
      expected = 0
      matches = @(
          '\bhelm\.sh/chart: "triton-distributed_worker-vllm"[\n\r]{1,2}'
          '\bapp\.kubernetes.io/instance: test[\n\r]{1,2}'
          '\bapp\.kubernetes.io/name: faux-triton[\n\r]{1,2}'
          '\s{8}- TRITON_MODEL_REPOSITORY: "/var/run/models"[\n\r]{1,2}'
          '\s{8}- TRITON_COMPONENT_NAME: "faux-triton"[\n\r]{1,2}'
          '\bimage: some_false-container_name:with_a-tag[\n\r]{1,2}'
          '\bephemeral-storage: 1Gi[\n\r]{1,2}'
          @{
            indent = 0
            lines = @(
              'apiVersion: apps/v1'
              'kind: Deployment'
              'metadata:'
              '  name: test'
            )
          }
        )
      options = @()
      values = @('basic_values.yaml')
    }
    @{
      name = 'basic_error'
      expected = 1
      matches = @(
          'Error: values don''t meet the specifications of the schema\(s\) in the following chart\(s\):'
          'triton-distributed_worker-vllm:'
          '- triton: componentName is required[\n\r]{1,2}'
          '- image: name is required[\n\r]{1,2}'
        )
      options = @()
      values = @()
    }
    @{
      name = 'volume_mounts'
      expected = 0
      matches = @(
          @{
            indent = 6
            lines = @(
              'volumes:'
              '- name: mount_w_path'
              '  persistentVolumeClaim:'
              '    claimName: w_path_pvc'
              '- name: mount_wo_path'
              '  persistentVolumeClaim:'
              '    claimName: wo_path_pvc'
              '- name: shared-memory'
              '  emptyDir:'
              '    medium: Memory'
              '    sizeLimit: 512Mi'
            )
          }
          @{
            indent = 8
            lines = @(
              'volumeMounts:'
              '- mountPath: /var/run/models/subpath'
              '  name: mount_w_path'
              '- mountPath: /var/run/models'
              '  name: mount_wo_path'
              '- mountPath: /dev/shm'
              '  name: shared-memory'
            )
          }
        )
      options = @()
      values = @(
          'basic_values.yaml'
          'volume_mounts.yaml'
        )
    }
    @{
      name = 'bad_volume_mounts'
      expected = 1
      matches = @(
          '- modelRepository.volumeMounts.0: persistentVolumeClaim is required\s[\n\r]{1,2}'
        )
      options = @()
      values = @(
          'basic_values.yaml'
          'bad_volume_mounts.yaml'
        )
    }
  )

$config = initialize_test $componentName $componentType $args $tests

# Being w/ the state of not having passed.
$is_pass = $false

try {
  $is_pass = $(test_helm_chart $config)
  write-debug "is_pass: ${is_pass}."
}
catch {
  if ($(get_is_debug)) {
    throw $_
  }

  fatal_exit "$_"
}

# Clean up any NVBUILD environment variables left behind by the build.
cleanup_after

if (-not $is_pass) {
  exit -1
}

exit 0
