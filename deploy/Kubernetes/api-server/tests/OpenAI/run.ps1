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
. "$(& git rev-parse --show-toplevel)/deploy/Kubernetes/_build/helm-test.ps1"

$componentName = 'api-server'
$componentType = 'OpenAI'

$tests = @(
    @{
      name = 'basic'
      expected = 0
      matches = @(
          '\s{8}helm\.sh\/chart:\s+"triton-distributed_api-server-openai"\s*[\n\r]{1,2}'
          '\s{8}app\.kubernetes\.io/instance:\s+test\s*[\n\r]{1,2}'
          '\s{8}app\.kubernetes\.io/name:\s+triton-distributed_api-server\s*[\n\r]{1,2}'
          '\s{8}image:\s+triton-distributed_image-name\s*[\n\r]{1,2}'
          '\s{8}- containerPort:\s+8000\s*[\n\r]{1,2}\s{10}name:\s+health\s*[\n\r]{1,2}'
          '\s{8}- containerPort:\s+9345\s*[\n\r]{1,2}\s{10}name:\s+request\s*[\n\r]{1,2}'
          '\s{8}- containerPort:\s+443\s*[\n\r]{1,2}\s{10}name:\s+api\s*[\n\r]{1,2}'
          '\s{8}- containerPort:\s+9347\s*[\n\r]{1,2}\s{10}name:\s+metrics\s*[\n\r]{1,2}'
        )
      options = @()
      values = @('basic.yaml')
    }
    @{
      name = 'basic_error'
      expected = 1
      matches = @(
          'Error: values don''t meet the specifications of the schema\(s\) in the following chart\(s\):\s*[\n\r]{1,2}'
          '- triton: componentName is required\s*[\n\r]{1,2}'
          '- image: name is required\s*[\n\r]{1,2}'
        )
      options = @()
      values = @()
    }
    @{
      name= 'kubernetes_correct'
      expected = 0
      matches = @(
        @{
          indent = 2
          lines = @(
            'annotations:'
            '  helm.sh/chart: "triton-distributed_api-server-openai"'
            '  triton-distributed: "test\.1\.0\.0"'
            '  random_thing: just-a-value'
            '  thing_random: another-item'
          )
        }
        @{
          indent = 2
          lines = @(
            'labels:'
            '  app: test'
            '  app\.kubernetes\.io/component: api-server'
            '  app\.kubernetes\.io/instance: test'
            '  app\.kubernetes\.io/name: triton-distributed_api-server'
            '  app\.kubernetes\.io/part-of: test_harness'
          )
        }
        @{
          indent = 6
          lines = @(
            'tolerations:'
            '- effect: NoSchedule'
            '  key: faux-taint'
            '  operator: Exists'
          )
        }
        @{
          indent = 8
          lines = @(
            'ports:'
            '- containerPort: 8000'
            '  name: health'
            '- containerPort: 9345'
            '  name: request'
            '- containerPort: 443'
            '  name: api'
            '- containerPort: 9347'
            '  name: metrics'
          )
        }
        @{
          indent = 0
          lines = @(
            'kind: Service'
            'apiVersion: v1'
            'metadata:'
            '  name: "triton_api-server"'
            '  namespace: "default"'
          )
        }
        @{
          indent = 0
          lines = @(
            'apiVersion: apps/v1'
            'kind: Deployment'
            'metadata:'
            '  name: test'
            '  namespace: "default"'
          )
        }
      )
      options = @()
      values = @(
        'basic.yaml'
        'kube_correct.yaml'
      )
    }
    @{
      name= 'kubernetes_invalid'
      expected = 1
      matches = @(
        'Error: values don''t meet the specifications of the schema\(s\) in the following chart\(s\):\s*[\n\r]{1,2}'
        'triton-distributed_api-server-openai:\s*[\n\r]{1,2}'
        '- kubernetes\.annotations: Invalid type. Expected: object, given: array\s*[\n\r]{1,2}'
        '- kubernetes\.labels: Invalid type. Expected: object, given: array\s*[\n\r]{1,2}'
        '- kubernetes\.tolerations\.0\.effect: Does not match pattern ''\^NoExecute\|NoSchedule\|PreferNoSchedule\$''\s*[\n\r]{1,2}'
        '- kubernetes\.tolerations\.0\.key: Invalid type. Expected: string, given: integer\s*[\n\r]{1,2}'
        '- kubernetes\.tolerations\.0\.operator: Does not match pattern ''\^Exists\|Equals\$''\s*[\n\r]{1,2}'
      )
      options = @()
      values = @(
        'basic.yaml'
        'kube_invalid.yaml'
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
