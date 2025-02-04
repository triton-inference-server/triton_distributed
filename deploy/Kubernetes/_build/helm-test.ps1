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

function initialize_test([string] $component_kind, [string] $component_name, [string[]]$params, [object[]] $tests) {
  write-debug "<initialize_test> component_kind: ${component_kind}."
  write-debug "<initialize_test> component_name: ${component_name}."
  write-debug "<initialize_test> params.count: $($params.count)."
  write-debug "<initialize_test> tests.count: $($tests.count)."

  $is_debug = $false
  $test_filter = @()

  for ($i = 0 ; $i -lt $params.count ; $i += 1) {
    $arg = $params[$i]

    if ('--debug' -ieq $arg) {
      $is_debug = $true
    }
    elseif ('--list' -ieq $arg)
    {
      write-minimal "Available tests:"

      foreach ($test in $tests) {
        write-minimal "- $($test.name)"
      }

      exit(0)
    }
    elseif (('--test' -ieq $arg) -or ('-t' -ieq $arg))
    {
      if ($i + 1 -ge $params.count) {
        usage_exit "Expected value following ""{$arg}""."
      }

      $i += 1
      $test_name = $params[$i]
      $test_found = $false

      $parts = $test_name.split('/')
      if ($parts.count -gt 1) {
        $test_name = $parts[$parts.count - 1]
      }

      foreach ($test in $tests) {
        if ($test.name -ieq $test_name) {
          $test_found = $true
          break
        }
      }

      if (-not $test_found) {
        usage_exit "Unknown test name ""${test_name}"" provided."
      }

      $test_filter += $test_name
    }
    elseif (('--verbose' -ieq $arg) -or ('-v' -ieq $arg)) {
      set_verbosity('DETAILED')
    }
    else {
      usage_exit "Unknown option '${arg}'."
    }
  }

  $is_debug = $is_debug -or $(is_debug)
  if ($is_debug) {
    $DebugPreference = 'Continue'
  }
  else {
    $DebugPreference = 'SilentlyContinue'
  }

  if (-not ($(get_verbosity) -eq 'MINIMAL' -and $(is_tty))) {
    set_verbosity('DETAILED')
  }

  # When a subset of tests has been requested, filter out the not requested tests.
  if ($test_filter.count -gt 0) {
    write-debug "<test-chart> selected.count: $($test_filter.count)."

    $replace = @()

    # Find the test that matches each selected item and add it to a replacement list.
    foreach ($filter in $test_filter) {
      foreach ($test in $tests) {
        if ($test.name -ieq $filter) {
          $replace += $test
          break
        }
      }
    }

    # Replace the test list with the replacement list.
    $tests = $replace
  }

  return @{
    component = $component_kind
    name = $component_name
    is_debug = $is_debug
    tests = $tests
  }
}

function test_helm_chart([object] $config) {
  write-debug "<test_helm_chart> config.component = '$($config.component)'."
  write-debug "<test_helm_chart> config.name = '$($config.name)'."
  write-debug "<test_helm_chart> config.count = [$($config.tests.count)]"

  $chart_path = to_local_path "deploy/Kubernetes/$($config.component)/charts/$($config.name)"
  $tests_path = to_local_path "deploy/Kubernetes/$($config.component)/tests/$($config.name)"

  push-location $chart_path

  try {
    $fail_count = 0
    $pass_count = 0

    foreach ($test in $config.tests) {
      $helm_command = "helm template test -f $(resolve-path './values.yaml' -relative)"
      write-debug "<test_helm_chart> helm_command = '${helm_command}'."

      # First add all values files to the command.
      if (($null -ne $test.values) -and ($test.values.count -gt 0)) {
        foreach ($value in $test.values) {
          write-debug "<test_helm-chart> value = '${value}'."
          $helm_command = "${helm_command} -f $(resolve-path "${tests_path}/${value}" -relative)"
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
        write-passed "$($config.component)/$($config.name)/$($test.name)"
      }
      else {
        $fail_count += 1
        write-failed "$($config.component)/$($config.name)/$($test.name)"
        write-low "  command: ${helm_command}"
      }
    }
  }
  catch {
    pop-location

    throw $_
  }

  pop-location

  if ($fail_count -gt 0) {
    write-minimal "Failed: ${fail_count}, Passed: ${pass_count}, Total: $($config.tests.count)" 'Red'
    return $false
  }
  else
  {
    write-minimal "Passed: ${pass_count}, Total: $($config.tests.count)" 'Green'
    return $true
  }
}
