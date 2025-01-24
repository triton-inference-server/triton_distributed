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

. "$(& git rev-parse --show-toplevel)/deploy/Kubernetes/_build/common.ps1"

function test_helm_chart([string] $chart_path, [string] $tests_path, [object[]] $test_set) {
  write-debug "<test_helm_chart> chart_path = '${chart_path}'."
  write-debug "<test_helm_chart> tests_path = '${tests_path}'."
  write-debug "<test_helm_chart> test_set = [$($test_set.count)]"

  $chart_path = to_local_path $chart_path
  $tests_path = to_local_path $tests_path

  push-location $chart_path

  try {
    $fail_count = 0
    $pass_count = 0

    foreach ($test in $test_set) {
      $helm_command = 'helm template test -f ./values.yaml'
      write-debug "<test_helm_chart> helm_command = '${helm_command}'."

      # First add all values files to the command.
      if (($null -ne $test.values) -and ($test.values.count -gt 0)) {
        foreach ($value in $test.values) {
          write-debug "<test_helm-chart> value = '${value}'."
          $helm_command = "${helm_command} -f ${tests_path}/${value}"
        }
        write-debug "<test_helm_chart> helm_command = '${helm_command}'."
      }

      # Second add all --set options to the command.
      if (($null -ne $test.options) -and $($test.options.count -gt 0)) {
        foreach ($option in $test.options) {
          write-debug "<test_helm_chart> option = '${option}'."
          $helm_command = "${helm_command} --set `"${option}`""
        }
      }

      $helm_command = "${helm_command} ."
      write-debug "<test_helm_chart> helm_command = '${helm_command}'."

      $captured = invoke-expression "${helm_command} 2>&1" | out-string
      $exit_code = $LASTEXITCODE
      write-debug "<test_helm_chart> expected = $($test.expected)."
      write-debug "<test_helm_chart> actual = ${exit_code}."

      $is_pass = $test.expected -eq $exit_code

      if (-not $is_pass) {
        write-low ">> Helm exited w/ ${exit_code}, test expected $($test.expected)."
      }

      if (($null -ne $test.matches) -and ($test.matches.count -gt 0)) {
        foreach ($match in $test.matches) {
          write-debug "<test_helm_chart> match = '${match}'."
          $is_match = $captured -match $match
          write-debug "<test_helm_chart> is_match = ${is_match}."

          if (-not $is_match) {
            write-low ">> Failed to match expected output: '${match}'."
          }

          $is_pass = $is_pass -and $is_match
        }
      }

      if ($is_pass) {
        $pass_count += 1
        write-passed "$($test.name)"
      }
      else {
        $fail_count += 1
        write-failed "$($test.name)"
      }
    }
  }
  catch {
    pop-location

    throw $_
  }

  pop-location

  if ($fail_count -gt 0) {
    write-minimal "Failed: ${fail_count}, Passed: ${pass_count}, Total: $($tests.count)" 'Red'
    return $false
  }
  else
  {
    write-minimal "Passed: ${pass_count}, Total: $($tests.count)" 'Green'
    return $true
  }
}
